import { NextResponse } from 'next/server';
import { TARGET_SERVER_BASE_URL } from '@/utils/serverConfig';

// The target backend server base URL, derived from environment variable or defaulted.

export async function GET(request: import('next/server').NextRequest) {
  try {
    const authorizationHeader = request.headers.get('Authorization');
    const targetUrl = `${TARGET_SERVER_BASE_URL}/models/config`;

    const backendHeaders: HeadersInit = {
      'Accept': 'application/json',
    };
    if (authorizationHeader) {
      backendHeaders['Authorization'] = authorizationHeader;
    }

    // Make the actual request to the backend service
    const backendResponse = await fetch(targetUrl, {
      method: 'GET',
      headers: backendHeaders,
      credentials: 'include',
    });

    // If the backend service responds with an error
    if (!backendResponse.ok) {
       const errorBody = await backendResponse.text();
       console.error(`Backend server returned ${backendResponse.status}: ${errorBody}`);
      return NextResponse.json(
         { error: `Backend server returned ${backendResponse.status}`, details: errorBody },
        { status: backendResponse.status }
      );
    }

    // Forward the response from the backend
    const modelConfig = await backendResponse.json();
    return NextResponse.json(modelConfig);
  } catch (error) {
    console.error('Error fetching model configurations:', error);    
    return new NextResponse(JSON.stringify({ error: error }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      });
  }
}

// Handle OPTIONS requests for CORS if needed
export function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
}
