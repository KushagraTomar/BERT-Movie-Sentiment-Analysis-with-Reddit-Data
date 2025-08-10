import pytest
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import create_app

@pytest.fixture
def app():
    """Create application for testing"""
    app = create_app('testing')
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/api/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'
    assert 'timestamp' in data
    assert 'model_loaded' in data
    assert 'reddit_available' in data

def test_predict_endpoint_missing_text(client):
    """Test predict endpoint with missing text"""
    response = client.post('/api/predict', 
                          json={},
                          content_type='application/json')
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Text is required' in data['error']

def test_predict_endpoint_empty_text(client):
    """Test predict endpoint with empty text"""
    response = client.post('/api/predict', 
                          json={'text': ''},
                          content_type='application/json')
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Text cannot be empty' in data['error']

def test_predict_endpoint_valid_text(client):
    """Test predict endpoint with valid text"""
    response = client.post('/api/predict', 
                          json={'text': 'This is a great movie!'},
                          content_type='application/json')
    
    # May return 503 if model not loaded, which is acceptable
    assert response.status_code in [200, 503]
    
    data = json.loads(response.data)
    if response.status_code == 200:
        assert 'success' in data
        assert 'result' in data
        assert 'timestamp' in data
    else:
        assert 'error' in data

def test_analyze_movie_missing_name(client):
    """Test analyze movie endpoint with missing movie name"""
    response = client.post('/api/analyze-movie', 
                          json={},
                          content_type='application/json')
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Movie name is required' in data['error']

def test_analyze_movie_empty_name(client):
    """Test analyze movie endpoint with empty movie name"""
    response = client.post('/api/analyze-movie', 
                          json={'movie_name': ''},
                          content_type='application/json')
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Movie name cannot be empty' in data['error']

def test_analyze_movie_valid_name(client):
    """Test analyze movie endpoint with valid movie name"""
    response = client.post('/api/analyze-movie', 
                          json={'movie_name': 'Test Movie'},
                          content_type='application/json')
    
    # May return 503 if model or Reddit not available, which is acceptable
    assert response.status_code in [200, 503]
    
    data = json.loads(response.data)
    if response.status_code == 200:
        assert 'success' in data
        assert 'movie_name' in data
        assert 'total_texts' in data
    else:
        assert 'error' in data

def test_model_info_endpoint(client):
    """Test model info endpoint"""
    response = client.get('/api/model-info')
    
    # May return 503 if model not loaded, which is acceptable
    assert response.status_code in [200, 503]
    
    data = json.loads(response.data)
    if response.status_code == 200:
        assert 'success' in data
        assert 'model_info' in data
    else:
        assert 'error' in data

def test_404_endpoint(client):
    """Test 404 error handling"""
    response = client.get('/api/nonexistent')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Endpoint not found' in data['error']

def test_index_page(client):
    """Test main index page"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Movie Sentiment Analyzer' in response.data

if __name__ == '__main__':
    pytest.main([__file__])