import { GET, DELETE } from './route';
import { NextRequest } from 'next/server';
// TARGET_SERVER_BASE_URL is not used in this file, it uses PYTHON_BACKEND_URL internally
// For consistency in tests, we can define it or mock the internal constant if needed.
// However, the tests will mock fetch directly based on expected URL patterns.
const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_HOST || 'http://localhost:8001';
const PROJECTS_API_ENDPOINT = `${PYTHON_BACKEND_URL}/api/processed_projects`;
const CACHE_API_ENDPOINT = `${PYTHON_BACKEND_URL}/api/wiki_cache`;


// Mock the global fetch function
global.fetch = jest.fn();

describe('/api/wiki/projects route', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockClear();
  });

  // Tests for GET handler
  describe('GET handler', () => {
    it('should return 200 and projects data on successful backend fetch', async () => {
      const mockProjects = [{ id: '1', name: 'Project X' }];
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockProjects,
      });

      const req = new NextRequest('http://localhost/api/wiki/projects');
      const response = await GET(req);
      const body = await response.json();

      expect(response.status).toBe(200);
      expect(body).toEqual(mockProjects);
      expect(global.fetch).toHaveBeenCalledWith(
        PROJECTS_API_ENDPOINT,
        expect.objectContaining({
          method: 'GET',
          credentials: 'include',
          cache: 'no-store',
          headers: expect.objectContaining({'Content-Type': 'application/json'}),
        })
      );
    });

    it('should forward Authorization header for GET', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true, status: 200, json: async () => [],
      });
      const req = new NextRequest('http://localhost/api/wiki/projects', {
        headers: { Authorization: 'Bearer testtoken' },
      });
      await GET(req);
      expect(global.fetch).toHaveBeenCalledWith(
        PROJECTS_API_ENDPOINT,
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer testtoken' }),
        })
      );
    });

    it('should handle backend error for GET', async () => {
      const errorBody = { error: "Backend GET error" };
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false, status: 500, json: async () => errorBody, text: async () => JSON.stringify(errorBody)
      });
      const req = new NextRequest('http://localhost/api/wiki/projects');
      const response = await GET(req);
      const body = await response.json();
      expect(response.status).toBe(500);
      expect(body).toEqual(errorBody);
    });

    it('should handle fetch failure for GET', async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network GET failed'));
      const req = new NextRequest('http://localhost/api/wiki/projects');
      const response = await GET(req);
      const body = await response.json();
      expect(response.status).toBe(503); // Service Unavailable
      expect(body.error).toContain('Failed to connect to the Python backend. Network GET failed');
    });
  });

  // Tests for DELETE handler
  describe('DELETE handler', () => {
    const mockDeleteBody = { owner: 'o', repo: 'r', repo_type: 't', language: 'l' };

    it('should return 200 and success message on successful backend delete', async () => {
      const mockSuccessResponse = { message: 'Project deleted successfully' };
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockSuccessResponse,
      });

      const req = new NextRequest('http://localhost/api/wiki/projects', {
        method: 'DELETE',
        body: JSON.stringify(mockDeleteBody),
        headers: { 'Content-Type': 'application/json' }
      });
      const response = await DELETE(req);
      const body = await response.json();

      expect(response.status).toBe(200);
      expect(body).toEqual(mockSuccessResponse);
      const expectedUrl = `${CACHE_API_ENDPOINT}?owner=o&repo=r&repo_type=t&language=l`;
      expect(global.fetch).toHaveBeenCalledWith(
        expectedUrl,
        expect.objectContaining({
          method: 'DELETE',
          credentials: 'include',
          headers: expect.objectContaining({'Content-Type': 'application/json'}),
        })
      );
    });

    it('should forward Authorization header for DELETE', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true, status: 200, json: async () => ({ message: 'Deleted' }),
      });
      const req = new NextRequest('http://localhost/api/wiki/projects', {
        method: 'DELETE',
        body: JSON.stringify(mockDeleteBody),
        headers: { Authorization: 'Bearer testtoken', 'Content-Type': 'application/json' },
      });
      await DELETE(req);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String), // URL can be complex due to params
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer testtoken' }),
        })
      );
    });

    it('should return 400 for invalid DELETE body', async () => {
      const req = new NextRequest('http://localhost/api/wiki/projects', {
        method: 'DELETE',
        body: JSON.stringify({ owner: 'o' }), // Incomplete body
        headers: { 'Content-Type': 'application/json' }
      });
      const response = await DELETE(req);
      const body = await response.json();
      expect(response.status).toBe(400);
      expect(body.error).toContain('Invalid request body');
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it('should handle backend error for DELETE', async () => {
      const errorBody = { error: "Backend DELETE error" };
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false, status: 500, json: async () => errorBody, text: async () => JSON.stringify(errorBody)
      });
       const req = new NextRequest('http://localhost/api/wiki/projects', {
        method: 'DELETE',
        body: JSON.stringify(mockDeleteBody),
        headers: { 'Content-Type': 'application/json' }
      });
      const response = await DELETE(req);
      const body = await response.json();
      expect(response.status).toBe(500);
      expect(body).toEqual(errorBody);
    });

    it('should handle fetch failure for DELETE', async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network DELETE failed'));
      const req = new NextRequest('http://localhost/api/wiki/projects', {
        method: 'DELETE',
        body: JSON.stringify(mockDeleteBody),
        headers: { 'Content-Type': 'application/json' }
      });
      const response = await DELETE(req);
      const body = await response.json();
      expect(response.status).toBe(500);
      expect(body.error).toContain('Failed to delete project: Network DELETE failed');
    });
  });
});
