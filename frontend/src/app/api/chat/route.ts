import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();

    const backendForm = new FormData();
    const message = formData.get("message");
    const image = formData.get("image");

    if (!message) {
      return NextResponse.json({ error: "message is required" }, { status: 400 });
    }

    backendForm.append("message", message as string);
    if (image && image instanceof Blob) {
      const file = image as File;
      backendForm.append("image", file, file.name ?? "upload");
    }

    const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";
    const apiKey = process.env.BACKEND_API_KEY ?? "";

    const response = await fetch(`${backendUrl}/api/agent/run`, {
      method: "POST",
      headers: { "X-API-Key": apiKey },
      body: backendForm,
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.ok ? 200 : response.status });
  } catch (err) {
    console.error("[CHAT ROUTE ERROR]", err);
    return NextResponse.json(
      { error: "Failed to reach backend", detail: String(err) },
      { status: 502 }
    );
  }
}
