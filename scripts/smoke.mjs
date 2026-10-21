// Checks a deployed site: the page, the old posture.html redirect, the default VRM, and the security headers.
const base = process.argv[2] ?? "https://bodybuddy.cechung.com";
const checks = [["/", "Desk Buddy"]];
let failed = false;
for (const [path, title] of checks) {
  const res = await fetch(base + path);
  const body = await res.text();
  const ok = res.ok && body.includes(`<title>${title}</title>`) && res.headers.get("content-security-policy")?.includes("cdn.jsdelivr.net");
  console.log(ok ? "ok  " : "FAIL", path, res.status);
  failed ||= !ok;
}
const old = await fetch(base + "/posture.html", { redirect: "manual" });
console.log(old.status === 301 ? "ok  " : "FAIL", "/posture.html ->", old.status, old.headers.get("location"));
failed ||= old.status !== 301;
const vrm = await fetch(base + "/template.vrm", { method: "HEAD" });
console.log(vrm.ok ? "ok  " : "FAIL", "/template.vrm", vrm.status, vrm.headers.get("content-length"));
failed ||= !vrm.ok;
process.exit(failed ? 1 : 0);
