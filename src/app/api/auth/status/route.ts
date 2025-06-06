import { NextRequest, NextResponse } from "next/server";
import { TARGET_SERVER_BASE_URL } from '@/utils/serverConfig';

export async function GET(request: NextRequest) {
  try {
    const authorizationHeader = request.headers.get('Authorization');
    const backendHeaders: HeadersInit = {};
    if (authorizationHeader) {
      backendHeaders['Authorization'] = authorizationHeader;
    }

    // Forward the request to the backend API
    const response = await fetch(`${TARGET_SERVER_BASE_URL}/auth/status`, {
      method: 'GET',
      headers: backendHeaders,
      credentials: 'include',
    });
    
    if (!response.ok) {
      const errorBody = await response.text();
      console.error(`Backend server returned ${response.status}: ${errorBody}`);
      return NextResponse.json(
        { error: `Backend server returned ${response.status}`, details: errorBody },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error forwarding request to backend:', error);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
