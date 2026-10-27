/*
 * Playground: one runner session for the open playground.
 *
 * Talks to the webui proxy (/api/playground/runner, docs/PLAYGROUND_RUNNER.md).
 * The runner runs one execution per session at a time and answers 409 to a
 * second, so every run goes through one queue here: a cell clicked while
 * another runs waits its turn instead of failing. Interrupt, Restart and Stop
 * drop whatever is still waiting.
 *
 * `ran` is the set of cells that ran without error in the current kernel. It
 * is emptied whenever the kernel's variables are lost (new session, restart,
 * stop, an expired session, or a kernel the runner had to restart), and
 * `generation` counts those losses so a result that arrives late cannot mark
 * a cell as run in a kernel that no longer exists.
 *
 * No DOM. fetch is injected, so node --test drives it with a fake
 * (tests/js/playground-session.test.js).
 */
(function (global) {
  'use strict';

  var Core = global.PlaygroundCore ||
    (typeof require === 'function' ? require('./playground-core.js') : {});

  var BASE = '/api/playground/runner';

  function Session(opts) {
    opts = opts || {};
    this.playgroundId = opts.playgroundId;
    // Run by the runner before any cell, on every start and restart.
    this.setupCode = opts.setupCode || '';
    // The runner's report on it: {status: 'none'|'ok'|'error', ename, evalue}.
    this.setup = null;
    this._fetch = opts.fetch || (typeof global.fetch === 'function' ? global.fetch.bind(global) : null);
    this.onChange = opts.onChange || function () {};
    this.phase = 'idle';
    this.message = '';
    this.id = null;
    this.limits = null;
    this.ran = new Set();
    this.generation = 0;
    this.waiting = 0;
    this.running = false;
    this._cancelSeq = 0;
    this._tail = Promise.resolve();
    this._starting = null;
  }

  Session.prototype._set = function (phase, message) {
    this.phase = phase;
    this.message = message || '';
    try { this.onChange(this); } catch (e) { /* a painter must not break the session */ }
  };

  Session.prototype._lost = function () {
    this.ran.clear();
    this.generation += 1;
  };

  Session.prototype._call = function (method, path, body, extra) {
    var init = { method: method, headers: { Accept: 'application/json' } };
    if (body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    if (extra && extra.keepalive) init.keepalive = true;
    if (!this._fetch) return Promise.resolve({ status: 0, body: { detail: 'No fetch.' } });
    return this._fetch(BASE + path, init).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        return { status: res.status, body: data || {} };
      });
    }, function (err) {
      return { status: 0, body: { detail: err && err.message } };
    });
  };

  Session.prototype._sid = function () {
    return '/sessions/' + encodeURIComponent(this.id);
  };

  Session.prototype._failed = function (res) {
    var detail = res.body && res.body.detail;
    if (res.status === 503) {
      this._set('unavailable', typeof detail === 'string' ? detail : 'The playground runner is not reachable.');
    }
    return Core.errorMessage(res.status, detail);
  };

  /** Remember the runner's setup report; a failed setup is the chip's hint. */
  Session.prototype._setupDone = function (body, okMessage) {
    this.setup = (body && body.setup) || null;
    var failed = Core.setupFailure(this.setup);
    this._set('ready', failed || okMessage || '');
  };

  /** Code to run before any cell from the next start or restart on. */
  Session.prototype.setSetupCode = function (code) {
    this.setupCode = String(code == null ? '' : code);
  };

  /** GET /health: the runner's limits, or null when it is not reachable. */
  Session.prototype.health = function () {
    var self = this;
    return this._call('GET', '/health').then(function (res) {
      if (res.status === 200) {
        self.limits = res.body.limits || null;
        if (self.phase === 'unavailable') self._set(self.id ? 'ready' : 'idle');
        return self.limits;
      }
      self._failed(res);
      return null;
    });
  };

  /**
   * The runner is full. A page reloaded while its kernel was still starting
   * never learns that session's id, so it cannot close it on the way out;
   * those leftovers would hold a slot until they idle out (30 minutes).
   * Close the least recently used idle session of this same playground and
   * say whether one was closed. Sessions of other playgrounds are left alone.
   */
  Session.prototype._reclaim = function () {
    var self = this;
    return this._call('GET', '/sessions').then(function (res) {
      if (res.status !== 200 || !Array.isArray(res.body.sessions)) return false;
      var spare = res.body.sessions.filter(function (s) {
        return s.playground_id === self.playgroundId && !s.busy && s.session_id !== self.id;
      }).sort(function (a, b) {
        return String(a.last_active_at).localeCompare(String(b.last_active_at));
      })[0];
      if (!spare) return false;
      return self._call('DELETE', '/sessions/' + encodeURIComponent(spare.session_id)).then(function (del) {
        return del.status === 200 || del.status === 404;
      });
    });
  };

  /** Start a kernel. Resolves true when there is a session to run in. */
  Session.prototype.start = function () {
    var self = this;
    if (this._starting) return this._starting;
    this._set('starting');
    var create = function () {
      return self._call('POST', '/sessions', { playground_id: self.playgroundId, setup_code: self.setupCode });
    };
    this._starting = create().then(function (res) {
      if (res.status !== 429) return res;
      return self._reclaim().then(function (freed) { return freed ? create() : res; });
    }).then(function (res) {
      self._starting = null;
      if (res.status === 201 || res.status === 200) {
        self.id = res.body.session_id;
        self._lost();
        self._setupDone(res.body);
        return true;
      }
      var message = self._failed(res);
      if (res.status !== 503) self._set('error', message);
      return false;
    });
    return this._starting;
  };

  Session.prototype.ensure = function () {
    return this.id ? Promise.resolve(true) : this.start();
  };

  Session.prototype.timeoutS = function () {
    return (this.limits && this.limits.exec_timeout_s) || 180;
  };

  Session.prototype.maxTimeoutS = function () {
    return (this.limits && this.limits.max_exec_timeout_s) || 900;
  };

  /**
   * Run code. Resolves to one of:
   *   {result, generation}  the runner answered (check result.status)
   *   {error, status}       it did not; error is a sentence for the page
   *   {cancelled: true}     dropped from the queue by Interrupt/Restart/Stop
   * `onStart` fires when this run leaves the queue and starts executing.
   */
  Session.prototype.execute = function (code, timeoutS, onStart) {
    var self = this;
    var seq = this._cancelSeq;
    this.waiting += 1;
    if (this.running) this._set('busy', this.waiting + ' waiting');
    var job = this._tail.then(function () {
      if (seq !== self._cancelSeq) return { cancelled: true };
      return self.ensure().then(function (ok) {
        if (!ok) return { error: self.message || 'The session could not start.', status: 503 };
        if (seq !== self._cancelSeq) return { cancelled: true };
        // The runner strips a leading indent the way Jupyter did
        // (playground_runner/indent.py), so the code goes as written.
        var body = { code: String(code == null ? '' : code) };
        if (timeoutS) body.timeout_s = timeoutS;
        var generation = self.generation;
        self.running = true;
        self._set('busy', self.waiting > 1 ? (self.waiting - 1) + ' waiting' : 'Running code.');
        if (onStart) { try { onStart(); } catch (e) { /* display only */ } }
        return self._call('POST', self._sid() + '/execute', body).then(function (res) {
          self.running = false;
          if (res.status === 200) {
            if (res.body.kernel_restarted) self._lost();
            self._set('ready');
            return { result: res.body, generation: res.body.kernel_restarted ? self.generation : generation };
          }
          if (res.status === 404) {
            self.id = null;
            self._lost();
            self._set('stopped', 'The session had ended, so its variables are gone.');
          } else {
            var msg = self._failed(res);
            if (res.status !== 503) self._set(self.id ? 'ready' : 'stopped');
            return { error: msg, status: res.status };
          }
          return { error: Core.errorMessage(404), status: 404 };
        });
      });
    });
    // A bug in this page must not leave the queue or a cell stuck on "Waiting".
    job = job.catch(function (err) {
      self.running = false;
      self._set(self.id ? 'ready' : 'stopped');
      return { error: 'The page hit an error: ' + ((err && err.message) || err), status: 0 };
    });
    var settle = function () { self.waiting -= 1; };
    job.then(settle, settle);
    this._tail = job.catch(function () {});
    return job;
  };

  /** Record that `index` ran cleanly, unless the kernel changed since. */
  Session.prototype.markRan = function (index, generation) {
    if (generation === this.generation) this.ran.add(index);
  };

  Session.prototype.hasRan = function (index) {
    return this.ran.has(index);
  };

  /** Stop the running code and drop whatever is waiting. */
  Session.prototype.interrupt = function () {
    this._cancelSeq += 1;
    if (!this.id || !this.running) return Promise.resolve(false);
    return this._call('POST', this._sid() + '/interrupt').then(function (res) {
      return res.status === 200 && !!res.body.interrupted;
    });
  };

  /** A fresh kernel: every variable gone. Starts one if there is none. */
  Session.prototype.restart = function () {
    var self = this;
    this._cancelSeq += 1;
    if (!this.id) return this.start();
    this._set('starting');
    return this._call('POST', this._sid() + '/restart', { setup_code: this.setupCode }).then(function (res) {
      if (res.status === 200) {
        self._lost();
        self._setupDone(res.body, 'Restarted. Every variable is gone.');
        return true;
      }
      if (res.status === 404) {
        self.id = null;
        return self.start();
      }
      var msg = self._failed(res);
      if (res.status !== 503) self._set('error', msg);
      return false;
    });
  };

  Session.prototype.stop = function () {
    var self = this;
    this._cancelSeq += 1;
    var id = this.id;
    this.id = null;
    this._lost();
    if (!id) {
      this._set('stopped');
      return Promise.resolve(true);
    }
    return this._call('DELETE', '/sessions/' + encodeURIComponent(id)).then(function (res) {
      self._set('stopped', res.status === 200 || res.status === 404 ? 'Stopped. Run a cell to start a new session.' : Core.errorMessage(res.status, res.body.detail));
      return res.status === 200 || res.status === 404;
    });
  };

  /** The kernel is gone without our asking (the page came back from the back/forward cache). */
  Session.prototype.markGone = function (message) {
    this.id = null;
    this._lost();
    this._cancelSeq += 1;
    this._set('stopped', message);
  };

  /** The page is going away: close the kernel without waiting for an answer. */
  Session.prototype.leave = function () {
    if (!this.id) return;
    var id = this.id;
    this.id = null;
    this._lost();
    this._cancelSeq += 1;
    this._call('DELETE', '/sessions/' + encodeURIComponent(id), undefined, { keepalive: true });
    this.phase = 'stopped';
  };

  var api = { Session: Session, BASE: BASE };
  global.PlaygroundSession = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
