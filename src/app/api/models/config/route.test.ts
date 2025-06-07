import { GET } from './route'; // Adjust if your handler is exported differently
import { NextRequest } from 'next/server';
import { TARGET_SERVER_BASE_URL } from '@/utils/serverConfig';

// Mock the global fetch function
global.fetch = jest.fn();

describe('/api/models/config GET handler', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockClear();
  });

  it('should return 200 and data on successful backend fetch', async () => {
    const mockBackendResponse = { models: ["gpt-3.5-turbo", "gpt-4"] };
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockBackendResponse,
      text: async () => JSON.stringify(mockBackendResponse),
    });

    // Use NextRequest directly as createMocks might not be necessary for simple GET
    const req = new NextRequest('http://localhost/api/models/config');

    const response = await GET(req); // Pass the NextRequest object
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body).toEqual(mockBackendResponse);
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(global.fetch).toHaveBeenCalledWith(
      `${TARGET_SERVER_BASE_URL}/models/config`,
      expect.objectContaining({
        method: 'GET',
        credentials: 'include',
        headers: expect.objectContaining({
          'Accept': 'application/json',
        }),
      })
    );
  });

  it('should forward Authorization header', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ models: [] }),
      text: async () => JSON.stringify({ models: [] }),
    });

    const req = new NextRequest('http://localhost/api/models/config', {
      headers: {
        Authorization: 'Bearer testtoken',
      },
    });

    await GET(req);

    expect(global.fetch).toHaveBeenCalledWith(
      `${TARGET_SERVER_BASE_URL}/models/config`,
      expect.objectContaining({
        headers: expect.objectContaining({
          'Authorization': 'Bearer testtoken',
          'Accept': 'application/json',
        }),
        credentials: 'include',
      })
    );
  });

  it('should return backend error response when backend fetch is not ok', async () => {
    const errorDetails = 'Backend config error';
    // (global.fetch as jest.Mock).mockResolvedValueOnce({ // Original mock before reset
    //   ok: false,
    //   status: 502,
    // });

    const req = new NextRequest('http://localhost/api/models/config');
    // const response = await GET(req); // Original call before reset
    // const body = await response.json(); // Original parse before reset

    // expect(response.status).toBe(502); // Original assertion

    // Reset and set up mock for the enhanced error logging which expects .text()
     (global.fetch as jest.Mock).mockReset();
     (global.fetch as jest.Mock).mockResolvedValueOnce({
       ok: false,
       status: 502,
       text: async () => errorDetails, // Mock .text() method
     });

    const response_updated = await GET(req); // Call again with updated mock
    const body_updated = await response_updated.json();

    expect(response_updated.status).toBe(502);
    expect(body_updated.error).toBe('Backend server returned 502');
    expect(body_updated.details).toBe(errorDetails);
  });

  it('should return 500 when fetch itself fails', async () => {
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network failure'));

    const req = new NextRequest('http://localhost/api/models/config');
    const response = await GET(req);
    const body = await response.json();

    expect(response.status).toBe(500);
    // The route stringifies the error object directly: JSON.stringify({ error: error })
    // When error is an Error instance, this results in { error: {} }
    expect(body.error).toEqual({});
  });
});
