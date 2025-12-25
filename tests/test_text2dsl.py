"""
Tests for Text2DSL module - Natural Language to DSL conversion
"""

import pytest
from unittest.mock import Mock, patch


class TestText2DSL:
    """Tests for Text2DSL converter."""
    
    @pytest.fixture
    def text2dsl(self):
        from orchestrator.text2dsl import Text2DSL
        return Text2DSL({})
    
    def test_docker_restart_command(self, text2dsl):
        """Test Docker restart command parsing."""
        dsl = text2dsl.nl_to_dsl("zrestartuj backend")
        
        assert dsl is not None
        assert dsl["action"] == "docker.restart"
        assert dsl["target"] == "backend"
    
    def test_docker_stop_command(self, text2dsl):
        """Test Docker stop command parsing."""
        dsl = text2dsl.nl_to_dsl("zatrzymaj frontend")
        
        assert dsl is not None
        assert dsl["action"] == "docker.stop"
        assert dsl["target"] == "frontend"
    
    def test_docker_logs_command(self, text2dsl):
        """Test Docker logs command parsing."""
        dsl = text2dsl.nl_to_dsl("pokaż logi backend")
        
        assert dsl is not None
        assert dsl["action"] == "docker.logs"
        assert dsl["target"] == "backend"
    
    def test_docker_status_command(self, text2dsl):
        """Test Docker status command parsing."""
        dsl = text2dsl.nl_to_dsl("status kontenerów")
        
        assert dsl is not None
        assert dsl["action"] == "docker.status"
    
    def test_vision_describe_command(self, text2dsl):
        """Test vision describe command parsing."""
        dsl = text2dsl.nl_to_dsl("co widzisz")
        
        assert dsl is not None
        assert dsl["action"] == "vision.describe"
    
    def test_vision_count_command(self, text2dsl):
        """Test vision count command parsing."""
        dsl = text2dsl.nl_to_dsl("ile osób widzisz")
        
        assert dsl is not None
        assert dsl["action"] == "vision.count"
        assert dsl["target"] == "osób"
    
    def test_vision_find_command(self, text2dsl):
        """Test vision find command parsing."""
        dsl = text2dsl.nl_to_dsl("gdzie jest kubek")
        
        assert dsl is not None
        assert dsl["action"] == "vision.find"
        assert dsl["target"] == "kubek"
    
    def test_sensor_temperature_command(self, text2dsl):
        """Test sensor temperature command parsing."""
        dsl = text2dsl.nl_to_dsl("jaka jest temperatura")
        
        assert dsl is not None
        assert dsl["action"] == "sensor.read"
        assert dsl["metric"] == "temperature"
    
    def test_device_light_on_command(self, text2dsl):
        """Test device light on command parsing."""
        dsl = text2dsl.nl_to_dsl("włącz światło w kuchni")
        
        assert dsl is not None
        assert dsl["action"] == "device.set"
        assert dsl["device"] == "light"
        assert dsl["state"] == "on"
        assert dsl["location"] == "kuchni"
    
    def test_system_help_command(self, text2dsl):
        """Test system help command parsing."""
        dsl = text2dsl.nl_to_dsl("pomoc")
        
        assert dsl is not None
        assert dsl["action"] == "system.help"
    
    def test_system_exit_command(self, text2dsl):
        """Test system exit command parsing."""
        dsl = text2dsl.nl_to_dsl("koniec")
        
        assert dsl is not None
        assert dsl["action"] == "system.exit"
    
    def test_unknown_command_returns_none(self, text2dsl):
        """Test that unknown commands return None."""
        dsl = text2dsl.nl_to_dsl("zrób mi kanapkę")
        
        assert dsl is None
    
    def test_empty_input_returns_none(self, text2dsl):
        """Test that empty input returns None."""
        assert text2dsl.nl_to_dsl("") is None
        assert text2dsl.nl_to_dsl("   ") is None
        assert text2dsl.nl_to_dsl(None) is None


class TestDSLToNL:
    """Tests for DSL to Natural Language conversion."""
    
    @pytest.fixture
    def text2dsl(self):
        from orchestrator.text2dsl import Text2DSL
        return Text2DSL({})
    
    def test_docker_restart_success_response(self, text2dsl):
        """Test Docker restart success response."""
        dsl = {
            "action": "docker.restart",
            "target": "backend",
            "status": "ok"
        }
        
        response = text2dsl.dsl_to_nl(dsl)
        
        assert "backend" in response
        assert "zrestartowany" in response.lower() or "pomyślnie" in response.lower()
    
    def test_docker_restart_error_response(self, text2dsl):
        """Test Docker restart error response."""
        dsl = {
            "action": "docker.restart",
            "target": "backend",
            "status": "error",
            "error": "container not found"
        }
        
        response = text2dsl.dsl_to_nl(dsl)
        
        assert "nie udało" in response.lower() or "błąd" in response.lower()
    
    def test_vision_describe_response(self, text2dsl):
        """Test vision describe response."""
        dsl = {
            "action": "vision.describe",
            "status": "ok",
            "description": "Widzę kubek i laptop"
        }
        
        response = text2dsl.dsl_to_nl(dsl)
        
        assert "kubek" in response or "laptop" in response
    
    def test_system_help_response(self, text2dsl):
        """Test system help response."""
        dsl = {
            "action": "system.help",
            "status": "ok"
        }
        
        response = text2dsl.dsl_to_nl(dsl)
        
        assert len(response) > 50  # Help text should be substantial


class TestLLMPromptGeneration:
    """Tests for LLM prompt generation."""
    
    @pytest.fixture
    def text2dsl(self):
        from orchestrator.text2dsl import Text2DSL
        return Text2DSL({})
    
    def test_llm_prompt_contains_user_text(self, text2dsl):
        """Test that LLM prompt contains user text."""
        prompt = text2dsl.get_llm_prompt("uruchom serwer")
        
        assert "uruchom serwer" in prompt
    
    def test_llm_prompt_contains_available_actions(self, text2dsl):
        """Test that LLM prompt lists available actions."""
        prompt = text2dsl.get_llm_prompt("test")
        
        assert "docker" in prompt.lower()
        assert "sensor" in prompt.lower()
    
    def test_parse_llm_response_valid_json(self, text2dsl):
        """Test parsing valid JSON from LLM response."""
        llm_response = 'Sure, here is the DSL: {"action": "docker.restart", "target": "api"}'
        
        dsl = text2dsl.parse_llm_response(llm_response)
        
        assert dsl is not None
        assert dsl["action"] == "docker.restart"
        assert dsl["target"] == "api"
    
    def test_parse_llm_response_invalid_json(self, text2dsl):
        """Test parsing invalid JSON from LLM response."""
        llm_response = "I don't understand that command"
        
        dsl = text2dsl.parse_llm_response(llm_response)
        
        assert dsl is None
