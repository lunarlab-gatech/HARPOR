// The account menu of the public site (site.py ACCOUNT, in the header of index.html, help.html, demos.html and
// privacy.html): the email, the Storage line, Privacy and Sign out, shown only while the user is signed in. This file
// also makes the page's one Supabase client, so every page shares the session that the app's sign-in stored.
// site.py loads static/vendor/supabase-2.118.0.js, then config.js (HARPOR_CONFIG), then this file. On the app supa.js
// comes next: it claims the client and draws the menu after its own sign-in (HARPOR_ACCOUNT). On the other pages this
// file draws it alone. The tunnel pages load none of these files and have no menu.
"use strict";
(function () {
  const C = typeof HARPOR_CONFIG === "undefined" ? null : HARPOR_CONFIG;
  if (!C) return;
  const $ = (sel) => document.querySelector(sel);
  const mib = (b) => Math.round(Number(b || 0) / 1048576).toLocaleString("en-US");
  let made = null, claimed = false, wired = false;

  // The page's client, made on the first call. supabase-js 2.118.0 starts the implicit flow by default; PKCE keeps the
  // tokens out of the URL. The default storage key comes from the project's URL, so all pages read one session.
  function client() {
    if (!made) made = supabase.createClient(C.url, C.publishableKey, { auth: { flowType: "pkce", persistSession: true,
      autoRefreshToken: true, detectSessionInUrl: true } });
    return made;
  }
  // supa.js takes the client this way, as it loads, so the end of the parse leaves the menu to it.
  function claim() {
    claimed = true;
    return client();
  }
  // The menu of a signed-in user. The email is the person icon's name for a screen reader and the menu's first line.
  // Sign out is for this browser only: a personal worker of the same account stays signed in. Then after() runs: the
  // app reloads, the other pages hide the menu.
  function show(user, after) {
    const menu = $("#harpor-account");
    $("#harpor-email").textContent = $("#harpor-who").textContent = user.email || "signed in";
    $("#harpor-signout").onclick = async () => {
      await client().auth.signOut({ scope: "local" }).catch(() => {});
      after();
    };
    menu.hidden = false;
    if (wired) return;
    wired = true;
    // The menu closes on a click elsewhere and on Escape, as a menu does. Escape from a line of the menu takes the
    // focus back to the person icon, so it does not fall to the page.
    document.addEventListener("click", (e) => { if (menu.open && !menu.contains(e.target)) menu.open = false; });
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape" || !menu.open) return;
      const inside = menu.contains(document.activeElement);
      menu.open = false;
      if (inside) $("#harpor-account > summary").focus();
    });
  }
  async function storage(access) {  // "Storage: 120 of 2,048 MB", for an approved account
    const r = await client().rpc("storage_used");
    $("#harpor-storage").textContent = r.error ? "" : `Storage: ${mib(r.data)} of ${mib(access.max_result_bytes)} MB`;
  }
  // The focus leaves the menu first (to the Help link), so it does not fall to the page. The Storage line goes too, as
  // the next account may not be approved.
  function hide() {
    const menu = $("#harpor-account");
    if (menu.contains(document.activeElement)) $("#helplink").focus();
    menu.open = false;
    menu.hidden = true;
    $("#harpor-storage").textContent = "";
  }
  async function fill() {  // the Storage line of an approved account
    const a = await client().from("access").select("approved,max_result_bytes").maybeSingle();
    if (a.data && a.data.approved) await storage(a.data);
  }
  // The pages without supa.js: the menu when a session is stored, else nothing. A sign-in or sign-out in another tab
  // (supabase-js tells the other tabs) shows or hides it. supabase-js also says SIGNED_IN when a tab gets the focus
  // back, so a menu that shows already stays as it is; the reads wait for the callback to end, as supabase-js asks.
  // In a frame (a click trap) it stays hidden, as the app shows nothing there.
  async function alone() {
    if (window.top !== window.self) return;
    const sb = client();
    let session = null;
    try { session = (await sb.auth.getSession()).data.session; } catch (e) { /* signed out */ }
    sb.auth.onAuthStateChange((event, s) => {
      if (event === "SIGNED_OUT") hide();
      else if (event === "SIGNED_IN" && s && $("#harpor-account").hidden) {
        show(s.user, hide);
        setTimeout(() => { fill().catch(() => {}); }, 0);
      }
    });
    if (!session) return;
    show(session.user, hide);
    await fill();
  }

  window.HARPOR_ACCOUNT = { client, claim, show, storage };
  // A browser fires DOMContentLoaded once it has run the page's scripts, so by then supa.js has claimed the app's menu.
  document.addEventListener("DOMContentLoaded", () => {
    if (!claimed) alone().catch(() => {});
  });
})();
