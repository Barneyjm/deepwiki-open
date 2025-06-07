import { GET } from './route'; // Adjust if your handler is exported differently
import { createMocks, RequestMethod } from 'node-mocks-http';
import { NextRequest } from 'next/server';
import { TARGET_SERVER_BASE_URL } from '@/utils/serverConfig'; // Ensure this path is correct

// Mock the global fetch function
global.fetch = jest.fn();

describe('/api/auth/status GET handler', () => {
  beforeEach(() => {
    // Reset mocks before each test
    (global.fetch as jest.Mock).mockClear();
  });

  it('should return 200 and data on successful backend fetch', async () => {
    const mockBackendResponse = { message: 'OK' };
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockBackendResponse,
      text: async () => JSON.stringify(mockBackendResponse), // For error logging consistency
    });

    const { req } = createMocks<NextRequest>({
      method: 'GET',
      url: '/api/auth/status',
    });

    const response = await GET(req);
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body).toEqual(mockBackendResponse);
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(global.fetch).toHaveBeenCalledWith(
      `${TARGET_SERVER_BASE_URL}/auth/status`,
      expect.objectContaining({
        method: 'GET',
        credentials: 'include',
        headers: expect.not.objectContaining({
          'Content-Type': 'application/json',
        }),
      })
    );
  });

  it('should forward Authorization header and not include Content-Type', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ message: 'OK' }),
      text: async () => JSON.stringify({ message: 'OK' }),
    });

    const { req } = createMocks<NextRequest>({
      method: 'GET',
      url: '/api/auth/status',
      headers: {
        Authorization: 'Bearer testtoken',
      },
    });

    await GET(req);

    expect(global.fetch).toHaveBeenCalledWith(
      `${TARGET_SERVER_BASE_URL}/auth/status`,
      expect.objectContaining({
        method: 'GET',
        credentials: 'include',
        headers: {
          Authorization: 'Bearer testtoken',
        },
      })
    );
    // Check that Content-Type is not in the headers sent to the backend
    const fetchOptions = (global.fetch as jest.Mock).mock.calls[0][1];
    expect(fetchOptions.headers['Content-Type']).toBeUndefined();
  });

  it('should return backend error response when backend fetch is not ok', async () => {
    const errorDetails = 'Backend processing error';
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ error: 'Bad Request', details: errorDetails }),
      text: async () => `Backend server returned 400: ${errorDetails}`,
    });

    const { req } = createMocks<NextRequest>({
      method: 'GET',
      url: '/api/auth/status',
    });

    const response = await GET(req);
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.error).toContain('Backend server returned 400');
    expect(body.details).toBe('Backend server returned 400: Backend processing error'); // Matching enhanced error logging
  });

  it('should return 500 when fetch itself fails (e.g., network error)', async () => {
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network failure'));

    const { req } = createMocks<NextRequest>({
      method: 'GET',
      url: '/api/auth/status',
    });

    const response = await GET(req);
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.error).toBe('Internal Server Error');
  });
});
