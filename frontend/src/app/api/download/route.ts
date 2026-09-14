import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const filename = searchParams.get("filename");

  if (!filename) {
    return NextResponse.json({ error: "filename is required" }, { status: 400 });
  }

  const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";
  const apiKey = process.env.BACKEND_API_KEY ?? "";

  try {
    const response = await fetch(`${backendUrl}/api/documents/download/${encodeURIComponent(filename)}`, {
      headers: { "X-API-Key": apiKey },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(error, { status: response.status });
    }

    const headers = new Headers();
    const contentType = response.headers.get("Content-Type");
    const contentDisposition = response.headers.get("Content-Disposition");
    if (contentType) headers.set("Content-Type", contentType);
    if (contentDisposition) headers.set("Content-Disposition", contentDisposition);

    const body = await response.blob();
    return new NextResponse(body, { status: 200, headers });
  } catch (err) {
    console.error("[DOWNLOAD ROUTE ERROR]", err);
    return NextResponse.json({ error: "Failed to reach backend" }, { status: 502 });
  }
}