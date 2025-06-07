import { POST } from './route'; // Adjust if your handler is exported differently
import { createMocks, RequestMethod } from 'node-mocks-http';
import { NextRequest } from 'next/server';
import { TARGET_SERVER_BASE_URL } from '@/utils/serverConfig';

// Mock the global fetch function
global.fetch = jest.fn();

describe('/api/auth/validate POST handler', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockClear();
  });

  const mockRequestBody = { token: 'test-token' };

  it('should return 200 and data on successful backend POST', async () => {
    const mockBackendResponse = { valid: true };
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockBackendResponse,
      text: async () => JSON.stringify(mockBackendResponse),
    });

    const { req } = createMocks<NextRequest>({
      method: 'POST',
      url: '/api/auth/validate',
      body: mockRequestBody, // node-mocks-http handles JSON stringification for 'body'
    });

    // Manually construct NextRequest as createMocks might not fully replicate it for body parsing
    const nextReq = new NextRequest(`http://localhost${req.url!}`, {
      method: req.method,
      headers: req.headers,
      body: JSON.stringify(mockRequestBody), // Ensure body is stringified for NextRequest
    });


    const response = await POST(nextReq);
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body).toEqual(mockBackendResponse);
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(global.fetch).toHaveBeenCalledWith(
      `${TARGET_SERVER_BASE_URL}/auth/validate`,
      expect.objectContaining({
        method: 'POST',
        credentials: 'include',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify(mockRequestBody),
      })
    );
  });

  it('should forward Authorization header on POST', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ valid: true }),
      text: async () => JSON.stringify({ valid: true }),
    });

    const customHeaders = {
      Authorization: 'Bearer testtoken',
      'Content-Type': 'application/json', // NextRequest will infer this from body if not set
    };

    const { req } = createMocks<NextRequest>({
      method: 'POST',
      url: '/api/auth/validate',
      headers: customHeaders,
      body: mockRequestBody,
    });

    const nextReq = new NextRequest(`http://localhost${req.url!}`, {
      method: req.method,
      headers: req.headers,
      body: JSON.stringify(mockRequestBody),
    });

    await POST(nextReq);

    expect(global.fetch).toHaveBeenCalledWith(
      `${TARGET_SERVER_BASE_URL}/auth/validate`,
      expect.objectContaining({
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer testtoken',
        },
        body: JSON.stringify(mockRequestBody),
      })
    );
  });

  it('should return backend error response when backend POST is not ok', async () => {
    const errorDetails = 'Invalid token';
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ error: 'Unauthorized', details: errorDetails }),
      text: async () => `Backend server returned 401: ${errorDetails}`,
    });

    const { req } = createMocks<NextRequest>({
      method: 'POST',
      url: '/api/auth/validate',
      body: mockRequestBody,
    });
     const nextReq = new NextRequest(`http://localhost${req.url!}`, {
      method: req.method,
      headers: req.headers,
      body: JSON.stringify(mockRequestBody),
    });


    const response = await POST(nextReq);
    const body = await response.json();

    expect(response.status).toBe(401);
    expect(body.error).toContain('Backend server returned 401');
    expect(body.details).toBe('Backend server returned 401: Invalid token');
  });

  it('should return 500 when fetch itself fails', async () => {
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network failure'));

    const { req } = createMocks<NextRequest>({
      method: 'POST',
      url: '/api/auth/validate',
      body: mockRequestBody,
    });
    const nextReq = new NextRequest(`http://localhost${req.url!}`, {
      method: req.method,
      headers: req.headers,
      body: JSON.stringify(mockRequestBody),
    });

    const response = await POST(nextReq);
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.error).toBe('Internal Server Error');
  });

  it('should handle errors if request.json() fails (e.g. malformed JSON)', async () => {
    // Create a request with a non-JSON parsable body
    const nextReq = new NextRequest('http://localhost/api/auth/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }, // Still claim it's JSON
      body: 'this is not json',
    });

    const response = await POST(nextReq);
    const body = await response.json();

    expect(response.status).toBe(500); // Or 400 depending on desired behavior, 500 for generic catch
    expect(body.error).toBe('Internal Server Error'); // Or a more specific parsing error
    expect(global.fetch).not.toHaveBeenCalled(); // Fetch should not be called if body parsing fails
  });
});
