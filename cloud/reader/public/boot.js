// Runs before first paint, the same as the script in the app's base.html.
// A separate file because the reader's content-security-policy forbids inline script.
(function () {
  var root = document.documentElement;
  var theme = 'paper';
  var mode = null;
  var density = 'comfortable';
  try {
    var storedTheme = localStorage.getItem('nlp:theme');
    var storedMode = localStorage.getItem('nlp:mode');
    var storedDensity = localStorage.getItem('nlp:density');
    if (storedTheme === 'paper' || storedTheme === 'cockpit' || storedTheme === 'journal') theme = storedTheme;
    if (storedMode === 'light' || storedMode === 'dark') mode = storedMode;
    if (storedDensity === 'compact' || storedDensity === 'comfortable') density = storedDensity;
  } catch (e) {}
  root.setAttribute('data-theme', theme);
  if (mode) root.setAttribute('data-mode', mode);
  else root.removeAttribute('data-mode');
  root.setAttribute('data-density', density);
})();
