"""
Shock2 Custom Exceptions
Centralized exception handling for the Shock2 system
"""

class Shock2Exception(Exception):
    """Base exception for all Shock2-related errors"""
    pass

class ConfigurationError(Shock2Exception):
    """Raised when there's a configuration error"""
    pass

class NeuralNetworkError(Shock2Exception):
    """Raised when there's an error with neural network operations"""
    pass

class DataCollectionError(Shock2Exception):
    """Raised when there's an error collecting data"""
    pass

class ContentGenerationError(Shock2Exception):
    """Raised when there's an error generating content"""
    pass

class PublishingError(Shock2Exception):
    """Raised when there's an error publishing content"""
    pass

class StealthModeError(Shock2Exception):
    """Raised when there's an error with stealth operations"""
    pass

class DatabaseError(Shock2Exception):
    """Raised when there's a database-related error"""
    pass

class APIError(Shock2Exception):
    """Raised when there's an API-related error"""
    pass

class ValidationError(Shock2Exception):
    """Raised when data validation fails"""
    pass
