export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { password } = req.body;
  const SECRET = process.env.SCAN_PASSWORD;
  const GH_PAT = process.env.GH_PAT;
  const GH_REPO = process.env.GH_REPO;

  // Debug - check env vars exist (don't expose values)
  if (!SECRET) return res.status(500).json({ error: 'SCAN_PASSWORD not set in Vercel env vars' });
  if (!GH_PAT) return res.status(500).json({ error: 'GH_PAT not set in Vercel env vars' });
  if (!GH_REPO) return res.status(500).json({ error: 'GH_REPO not set in Vercel env vars' });

  if (password !== SECRET) {
    return res.status(401).json({ error: 'Wrong password' });
  }

  try {
    const response = await fetch(
      `https://api.github.com/repos/${GH_REPO}/actions/workflows/scan.yml/dispatches`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${GH_PAT}`,
          'Accept': 'application/vnd.github.v3+json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ref: 'main' }),
      }
    );

    if (response.status === 204) {
      return res.status(200).json({ ok: true });
    } else {
      const data = await response.text();
      return res.status(500).json({ error: 'GitHub API error', status: response.status, detail: data });
    }
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
