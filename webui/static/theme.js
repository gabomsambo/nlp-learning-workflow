/* Appearance settings and toasts. The no-flash script in <head> applies
   localStorage before CSS; this file only wires the dialog and later updates. */
(function () {
  var THEMES = ["paper", "cockpit", "journal"];
  var MODES = ["system", "light", "dark"];
  var DENSITIES = ["comfortable", "compact"];

  function read(key) {
    try {
      return localStorage.getItem(key);
    } catch (e) {
      return null;
    }
  }

  function write(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch (e) {}
  }

  function apply(theme, mode, density) {
    var root = document.documentElement;
    if (THEMES.indexOf(theme) === -1) theme = "paper";
    if (DENSITIES.indexOf(density) === -1) density = "comfortable";
    root.setAttribute("data-theme", theme);
    root.setAttribute("data-density", density);
    if (mode === "light" || mode === "dark") root.setAttribute("data-mode", mode);
    else root.removeAttribute("data-mode");
  }

  function current() {
    var root = document.documentElement;
    var mode = root.getAttribute("data-mode");
    return {
      theme: root.getAttribute("data-theme") || "paper",
      mode: mode === "light" || mode === "dark" ? mode : "system",
      density: root.getAttribute("data-density") || "comfortable",
    };
  }

  function check(form, name, value) {
    var inputs = form.querySelectorAll('input[name="' + name + '"]');
    for (var i = 0; i < inputs.length; i++) {
      inputs[i].checked = inputs[i].value === value;
    }
  }

  function selected(form, name, fallback) {
    var input = form.querySelector('input[name="' + name + '"]:checked');
    return input ? input.value : fallback;
  }

  function initSettings() {
    var dialog = document.getElementById("settings-dialog");
    var form = document.getElementById("settings-form");
    if (!dialog || !form) return;

    function sync() {
      var state = current();
      check(form, "theme", state.theme);
      check(form, "mode", state.mode);
      check(form, "density", state.density);
    }

    document.querySelectorAll("[data-open-settings]").forEach(function (button) {
      button.addEventListener("click", function () {
        var drawer = document.getElementById("nav-drawer");
        if (drawer && drawer.open) drawer.close();
        sync();
        if (typeof dialog.showModal === "function") dialog.showModal();
      });
    });

    form.addEventListener("change", function () {
      var state = current();
      var theme = selected(form, "theme", state.theme);
      var mode = selected(form, "mode", state.mode);
      var density = selected(form, "density", state.density);
      if (THEMES.indexOf(theme) === -1) theme = "paper";
      if (MODES.indexOf(mode) === -1) mode = "system";
      if (DENSITIES.indexOf(density) === -1) density = "comfortable";
      apply(theme, mode, density);
      write("nlp:theme", theme);
      write("nlp:mode", mode);
      write("nlp:density", density);
    });

    dialog.querySelectorAll("[data-dialog-close]").forEach(function (button) {
      button.addEventListener("click", function () {
        dialog.close();
      });
    });
    sync();
  }

  function initMenu() {
    var dialog = document.getElementById("nav-drawer");
    var open = document.getElementById("menu-open");
    if (!dialog || !open) return;
    open.addEventListener("click", function () {
      open.setAttribute("aria-expanded", "true");
      if (typeof dialog.showModal === "function") dialog.showModal();
    });
    dialog.addEventListener("close", function () {
      open.setAttribute("aria-expanded", "false");
    });
    dialog.querySelectorAll("a, [data-dialog-close]").forEach(function (el) {
      el.addEventListener("click", function () {
        if (dialog.open) dialog.close();
      });
    });
  }

  function showToast(message, options) {
    var region = document.getElementById("toasts");
    if (!region || !message) return;
    var opts = options || {};
    var toast = document.createElement("div");
    toast.className = "toast" + (opts.kind ? " toast-" + opts.kind : "");
    var text = document.createElement("p");
    text.textContent = String(message);
    toast.appendChild(text);
    if (opts.href) {
      var link = document.createElement("a");
      link.href = opts.href;
      link.textContent = opts.linkLabel || "Open";
      toast.appendChild(link);
    }
    region.appendChild(toast);
    window.setTimeout(function () {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, opts.duration || 6000);
  }

  window.nlpToast = showToast;

  function boot() {
    initSettings();
    initMenu();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
