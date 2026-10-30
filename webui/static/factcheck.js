/*
 * Fact-check claim helpers shared by the Listen script view and the paper page.
 *
 * Pure functions over the claims `GET /api/podcast/{id}/factcheck` returns:
 * which bucket a claim is in, how it is worded, and the one-line summary. Both
 * pages count from `claimBucket`, so the paper page and Listen can never
 * disagree about what a verdict means. Moved out of listen.js unchanged.
 */
(function (global) {
  'use strict';

  var FACTCHECK_SEVERITY = { high: 3, medium: 2, low: 1 };

  var TONE_ICONS = { high: '▲', medium: '◆', low: '○', neutral: '○' };

  function claimTone(claim) {
    claim = claim || {};
    if (claim.verdict === 'supported') return 'supported';
    if (claim.verdict === 'not_checkable') return 'neutral';
    return FACTCHECK_SEVERITY[claim.severity] ? claim.severity : 'neutral';
  }

  /*
   * One claim, one bucket. The summary bar and the issues panel both count from
   * this, so they can never disagree about what a verdict means.
   *
   *   contradicted  the paper says something incompatible (high severity)
   *   misattributed the paper says something else, or the claim is not the
   *                 authors' to make (contradicted background, wrong attribution)
   *   not_in_paper  the paper does not address it, and the claim is about the paper
   *   overstated    the core is right and a detail is not
   *   outside       probably true of the world, absent from the paper
   *   unchecked     an opinion, a prediction, a hedge — not a checkable claim
   *   supported     the paper states it
   *   unclassified  a verdict this build does not know; shown, never folded away
   */
  function claimBucket(claim) {
    claim = claim || {};
    if (claim.verdict === 'supported') return 'supported';
    if (claim.verdict === 'not_checkable') return 'unchecked';
    if (claim.verdict === 'contradicted') {
      return claim.severity === 'high' ? 'contradicted' : 'misattributed';
    }
    if (claim.verdict === 'unsupported') {
      return claim.severity === 'low' ? 'outside' : 'not_in_paper';
    }
    if (claim.verdict === 'partially_supported') {
      return claim.error_type === 'wrong_attribution' ? 'misattributed' : 'overstated';
    }
    return 'unclassified';
  }

  function claimPresentation(claim) {
    claim = claim || {};
    var bucket = claimBucket(claim);
    if (bucket === 'supported') {
      return { tone: 'supported', icon: '✓', label: 'Supported by the paper' };
    }
    if (bucket === 'unchecked') {
      return { tone: 'neutral', icon: '○', label: 'Opinion · not checked' };
    }
    if (bucket === 'contradicted') {
      return { tone: 'high', icon: '▲', label: 'Contradicts the paper' };
    }
    if (bucket === 'misattributed') {
      return { tone: 'medium', icon: '◆', label: 'Not what the authors say' };
    }
    if (bucket === 'not_in_paper') {
      return { tone: 'high', icon: '▲', label: 'Not in the paper' };
    }
    if (bucket === 'overstated') {
      // Grey, not amber, when the engine called it low severity: an overstatement
      // the paper does not contradict is not the same as a wrong number. The
      // icon follows the tone, so colour and shape always agree.
      var tone = claim.severity === 'medium' ? 'medium' : 'low';
      return { tone: tone, icon: TONE_ICONS[tone], label: 'Overstated' };
    }
    if (bucket === 'outside') {
      return { tone: 'low', icon: '○', label: 'Outside the paper' };
    }
    return { tone: 'neutral', icon: '○', label: 'Not classified' };
  }

  var TONE_RANK = { high: 3, medium: 2, low: 1, neutral: 0, supported: -1 };

  function strongestClaim(claims) {
    var best = null;
    var bestRank = -2;
    (claims || []).forEach(function (claim) {
      var rank = TONE_RANK[claimTone(claim)];
      if (rank === undefined) rank = 0;
      if (rank > bestRank) {
        best = claim;
        bestRank = rank;
      }
    });
    return best;
  }

  function issueClaims(claims) {
    return (claims || []).filter(function (claim) {
      if (!claim) return false;
      var bucket = claimBucket(claim);
      // Supported claims and opinions are not issues. Anything else is — a claim
      // this build cannot classify is shown as unclassified, never dropped.
      return bucket !== 'supported' && bucket !== 'unchecked';
    }).slice().sort(function (a, b) {
      var severity = (FACTCHECK_SEVERITY[b.severity] || 0) - (FACTCHECK_SEVERITY[a.severity] || 0);
      if (severity) return severity;
      var line = Number(a.line_index) - Number(b.line_index);
      if (line) return line;
      return Number(a.span_start) - Number(b.span_start);
    });
  }

  var SUMMARY_LABELS = {
    contradicted: 'contradict the paper',
    not_in_paper: 'not in the paper',
    misattributed: 'not what the authors say',
    overstated: 'overstated',
    outside: 'outside the paper',
    unchecked: 'opinion (not checked)',
    unclassified: 'not classified',
    supported: 'supported',
  };

  var SUMMARY_ORDER = [
    'contradicted',
    'not_in_paper',
    'misattributed',
    'overstated',
    'outside',
    'unchecked',
    'unclassified',
    'supported',
  ];

  /*
   * The honest one-liner: how many claims were graded, and how they came out.
   *
   * Every bucket is counted, including the opinions that were not checked and
   * the claims the paper supports. A summary that only named the problems
   * would let the unmentioned lines read as verified, which is the point the
   * research makes about the implied-truth effect.
   */
  // Flags (not supported claims, not opinions) a reader has marked fine.
  function dismissedCount(claims) {
    return (claims || []).filter(function (claim) {
      return claim && claim.dismissed_at && canDismiss(claim);
    }).length;
  }

  // Only a flag can be dismissed. A supported claim or an opinion has nothing
  // to disagree with.
  function canDismiss(claim) {
    if (!claim || !claim.id) return false;
    var bucket = claimBucket(claim);
    return bucket !== 'supported' && bucket !== 'unchecked';
  }

  var DISMISSAL_NOTE_MAX = 1000;

  /*
   * The note is required: it is what turns a click into a label somebody can
   * later check the verifier against. The server enforces the same rule; this
   * only saves the round trip and words the refusal.
   */
  function checkDismissalNote(raw) {
    var note = String(raw == null ? '' : raw).trim();
    if (!note) return { ok: false, note: '', error: 'Say why this flag is fine. The note is kept with the dismissal.' };
    if (note.length > DISMISSAL_NOTE_MAX) {
      return { ok: false, note: note, error: 'Keep the note under ' + DISMISSAL_NOTE_MAX + ' characters (it is ' + note.length + ').' };
    }
    return { ok: true, note: note, error: '' };
  }

  function factcheckSummary(claims) {
    claims = claims || [];
    var counts = {};
    SUMMARY_ORDER.forEach(function (bucket) { counts[bucket] = 0; });
    claims.forEach(function (claim) {
      if (!claim) return;
      var bucket = claimBucket(claim);
      counts[bucket] = (counts[bucket] || 0) + 1;
    });
    var parts = [claims.length + ' claim' + (claims.length === 1 ? '' : 's') + ' checked'];
    SUMMARY_ORDER.forEach(function (bucket) {
      if (!counts[bucket]) return;
      parts.push(counts[bucket] + ' ' + SUMMARY_LABELS[bucket]);
    });
    // A dismissal is a person's opinion about a flag. The flag stays counted
    // above; this says how many of them somebody has disagreed with.
    var dismissed = dismissedCount(claims);
    if (dismissed) parts.push(dismissed + ' dismissed');
    return parts.join(' · ');
  }

  function isFactcheckStale(expectedHash, currentHash) {
    return !!(expectedHash && currentHash && expectedHash !== currentHash);
  }


  /*
   * Paper page: one episode's fact check, read from the same endpoints Listen
   * uses. `panelState` decides which of eight things the panel has to say, so
   * the page never has to infer "no issues" from the absence of an answer.
   *
   *   input.check   { status, data }  GET /api/podcast/{id}/factcheck (0 = network)
   *   input.run     { ok, run, detail }  latest podcast_factcheck run
   *   input.hash    { ok, value }  the script's current script_sha256
   *
   * kinds: running, none, failed, stale, done, migration, unavailable.
   * Only `done` carries claims; every other kind is a statement about why
   * there are none to show.
   */
  var ACTIVE_RUN = ['pending', 'running'];
  var BAD_RUN = ['failed', 'cancelled', 'interrupted'];

  function panelState(input) {
    input = input || {};
    var check = input.check || { status: 0, data: {} };
    var runResult = input.run || { ok: false, detail: '' };
    var hash = input.hash || { ok: false, value: '' };
    var data = check.data || {};
    var run = runResult.ok ? runResult.run || null : null;
    var ok = check.status >= 200 && check.status < 300;

    if (!runResult.ok && check.status === 404) {
      return {
        kind: 'unavailable', tone: 'error',
        message: 'Could not tell whether a fact check is missing, running, or failed.',
        detail: runResult.detail || '',
      };
    }
    if (run && ACTIVE_RUN.indexOf(run.status) !== -1) {
      return { kind: 'running', tone: '', message: 'A fact check is running. No claims have been graded yet.', run: run };
    }
    if (check.status === 503) {
      return {
        kind: 'migration', tone: 'error',
        message: 'Fact-check results are unavailable because a database migration is missing.',
        detail: data.detail || '',
      };
    }
    if (!ok && check.status !== 404) {
      return {
        kind: 'unavailable', tone: 'error',
        message: 'The fact check could not be loaded. This is not the same as having no issues.',
        detail: data.detail || '',
      };
    }
    if (check.status === 404) {
      if (run && run.status === 'failed') {
        return {
          kind: 'failed', tone: 'error',
          message: 'The fact check failed, so no claim in this script has been graded.',
          detail: run.error || 'No failure reason was recorded.',
        };
      }
      if (run && BAD_RUN.indexOf(run.status) !== -1) {
        return {
          kind: 'failed', tone: 'warn',
          message: 'The fact check was ' + run.status + '. No claim in this script has been graded.',
          detail: '',
        };
      }
      if (run && run.status === 'succeeded') {
        return {
          kind: 'unavailable', tone: 'error',
          message: 'The fact-check run finished, but its saved result could not be found.',
          detail: '',
        };
      }
      return { kind: 'none', tone: '', message: 'Not fact-checked yet. No claims have been graded.' };
    }
    if (hash.ok && isFactcheckStale(data.script_sha256, hash.value)) {
      return {
        kind: 'stale', tone: 'warn',
        message: 'This fact check is stale because the script changed after it was checked. Re-check it in Listen.',
      };
    }
    var caveats = [];
    if (!hash.ok) caveats.push('Could not confirm the script is unchanged since this check.');
    if (!runResult.ok) {
      caveats.push('The fact-check run history could not be read, so this may not be the latest check.');
    } else if (run && BAD_RUN.indexOf(run.status) !== -1) {
      var checked = Date.parse(data.created_at || '') || 0;
      var ran = Date.parse(run.created_at || '') || 0;
      if (ran >= checked) {
        caveats.push(
          run.status === 'failed'
            ? 'The latest re-check failed. These results are from the previous completed check.'
            : 'The latest re-check was ' + run.status + '. These results are from the previous completed check.',
        );
      }
    }
    return {
      kind: 'done', tone: caveats.length ? 'warn' : '', message: '', caveats: caveats,
      claims: Array.isArray(data.claims) ? data.claims : [],
    };
  }

  /*
   * The items the paper page lists: high severity only, not dismissed. The full
   * list lives in Listen. Order matches Listen's issues panel (line, then span).
   */
  function highSeverityItems(claims) {
    return issueClaims(claims).filter(function (claim) {
      return claim.severity === 'high' && !claim.dismissed_at;
    });
  }

  // Zero-based, like the engine's line_index and Listen's ?line=.
  function scriptLineHref(scriptId, lineIndex) {
    if (typeof scriptId !== 'string' || !scriptId) return null;
    var base = '/podcast/?script=' + encodeURIComponent(scriptId);
    if (typeof lineIndex !== 'number' || !Number.isInteger(lineIndex) || lineIndex < 0) return base;
    return base + '&line=' + lineIndex;
  }

  function truncateText(text, max) {
    var value = String(text == null ? '' : text).replace(/\s+/g, ' ').trim();
    if (value.length <= max) return value;
    return value.slice(0, Math.max(0, max - 1)).replace(/\s+\S*$/, '') + '…';
  }

  var api = {
    claimTone: claimTone,
    claimBucket: claimBucket,
    claimPresentation: claimPresentation,
    strongestClaim: strongestClaim,
    issueClaims: issueClaims,
    factcheckSummary: factcheckSummary,
    isFactcheckStale: isFactcheckStale,
    canDismiss: canDismiss,
    dismissedCount: dismissedCount,
    checkDismissalNote: checkDismissalNote,
    DISMISSAL_NOTE_MAX: DISMISSAL_NOTE_MAX,
    panelState: panelState,
    highSeverityItems: highSeverityItems,
    scriptLineHref: scriptLineHref,
    truncateText: truncateText,
    SUMMARY_LABELS: SUMMARY_LABELS,
    SUMMARY_ORDER: SUMMARY_ORDER,
  };
  global.Factcheck = global.Factcheck || {};
  Object.keys(api).forEach(function (key) { global.Factcheck[key] = api[key]; });
  if (typeof module !== 'undefined' && module.exports) module.exports = global.Factcheck;
})(typeof window !== 'undefined' ? window : globalThis);
