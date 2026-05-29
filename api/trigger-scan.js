export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { password } = req.body;
  const SECRET = process.env.SCAN_PASSWORD;

  if (!SECRET || password !== SECRET) {
    return res.status(401).json({ error: 'Wrong password' });
  }

  try {
    const response = await fetch(
      `https://api.github.com/repos/${process.env.GH_REPO}/actions/workflows/scan.yml/dispatches`,
      {
        method: 'POST',
        headers: {
          'Authorization': `token ${process.env.GH_PAT}`,
          'Accept': 'application/vnd.github.v3+json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ref: 'main' }),
      }
    );

    if (response.status === 204) {
      return res.status(200).json({ ok: true, message: 'Scan triggered — check back in ~15 minutes' });
    } else {
      const data = await response.text();
      return res.status(500).json({ error: 'GitHub API error', detail: data });
    }
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
