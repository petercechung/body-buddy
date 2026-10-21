// Checks a deployed site: both pages, the default VRM, and the security headers.
const base = process.argv[2] ?? "https://bodybuddy.cechung.com";
const checks = [
  ["/", "Squat Mirror"],
  ["/posture.html", "Desk Buddy"],
];
let failed = false;
for (const [path, title] of checks) {
  const res = await fetch(base + path);
  const body = await res.text();
  const ok = res.ok && body.includes(`<title>${title}</title>`) && res.headers.get("content-security-policy")?.includes("cdn.jsdelivr.net");
  console.log(ok ? "ok  " : "FAIL", path, res.status);
  failed ||= !ok;
}
const vrm = await fetch(base + "/template.vrm", { method: "HEAD" });
console.log(vrm.ok ? "ok  " : "FAIL", "/template.vrm", vrm.status, vrm.headers.get("content-length"));
failed ||= !vrm.ok;
process.exit(failed ? 1 : 0);
