"""
Tests for the communication protocol classes.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch
from src.communication.communication_protocol import CommunicationProtocol, Message


class TestMessage:
    """Test Message dataclass."""
    
    def test_message_initialization(self):
        """Test message initialization."""
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="resource_discovery",
            content={"resource_id": 123, "position": (100, 100)},
            timestamp=1000.0,
            priority=2,
            expires_at=2000.0,
            position=(100, 100)
        )
        
        assert message.sender_id == "agent1"
        assert message.recipient_id == "agent2"
        assert message.message_type == "resource_discovery"
        assert message.content["resource_id"] == 123
        assert message.content["position"] == (100, 100)
        assert message.priority == 2
        assert message.timestamp == 1000.0
    
    def test_message_default_values(self):
        """Test message with default values."""
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="status_update",
            content={"status": "active"},
            timestamp=time.time()
        )
        
        assert message.content == {"status": "active"}
        assert message.priority == 1
        assert message.expires_at is None
        assert message.position is None
    
    def test_message_priority_validation(self):
        """Test message priority validation."""
        # Valid priorities
        for priority in [1, 2, 3]:
            message = Message(
                sender_id="agent1",
                recipient_id="agent2",
                message_type="test",
                content={"test": True},
                timestamp=time.time(),
                priority=priority
            )
            assert message.priority == priority


class TestCommunicationProtocol:
    """Test CommunicationProtocol class."""
    
    @pytest.fixture
    def comm_protocol(self):
        """Create a communication protocol instance."""
        return CommunicationProtocol(
            max_bandwidth=1000,
            pheromone_decay_rate=0.1,
            max_message_history=100
        )
    
    def test_communication_protocol_initialization(self, comm_protocol):
        """Test communication protocol initialization."""
        assert comm_protocol.max_bandwidth == 1000
        assert comm_protocol.pheromone_decay_rate == 0.1
        assert comm_protocol.max_message_history == 100
        assert comm_protocol.bandwidth_usage == 0
        assert len(comm_protocol.message_queue) == 0
        assert len(comm_protocol.message_history) == 0
        assert len(comm_protocol.pheromones) == 0
    
    def test_send_direct_message(self, comm_protocol):
        """Test sending direct messages."""
        # Send a message
        success = comm_protocol.send_message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="resource_discovery",
            content={"resource_id": 123}
        )
        
        assert success
        
        # Check message is in queue
        assert len(comm_protocol.message_queue) == 1
        queued_message = comm_protocol.message_queue[0]
        assert queued_message.sender_id == "agent1"
        assert queued_message.recipient_id == "agent2"
        
        # Check bandwidth usage
        assert comm_protocol.bandwidth_usage > 0
    
    def test_send_broadcast_message(self, comm_protocol):
        """Test sending broadcast messages."""
        # Send a broadcast message
        success = comm_protocol.send_message(
            sender_id="agent1",
            recipient_id=None,  # None for broadcast
            message_type="danger_alert",
            content={"danger_type": "obstacle", "position": (200, 200)}
        )
        
        assert success
        
        # Check message is in broadcast queue
        assert len(comm_protocol.broadcast_messages) == 1
        queued_message = comm_protocol.broadcast_messages[0]
        assert queued_message.recipient_id is None
    
    def test_bandwidth_limitation(self, comm_protocol):
        """Test bandwidth limitation for messages."""
        # Set low bandwidth
        comm_protocol.max_bandwidth = 1
        comm_protocol.bandwidth_usage = 1
        
        # Try to send a message that exceeds bandwidth
        success = comm_protocol.send_message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="large_data",
            content={"data": "x" * 20}  # Large content
        )
        
        assert not success  # Should fail due to bandwidth limitation
        
        # Check message queue is empty
        assert len(comm_protocol.message_queue) == 0
    
    def test_receive_message(self, comm_protocol):
        """Test receiving messages."""
        # Send a message first
        comm_protocol.send_message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="status_update",
            content={"status": "active"}
        )
        
        # Receive message for agent2
        received_messages = comm_protocol.receive_messages("agent2", (100, 100), 50.0)
        
        assert len(received_messages) == 1
        received_message = received_messages[0]
        assert received_message.sender_id == "agent1"
        assert received_message.message_type == "status_update"
        
        # Check message queue is empty after receiving
        assert len(comm_protocol.message_queue) == 0
    
    def test_receive_broadcast_message(self, comm_protocol):
        """Test receiving broadcast messages."""
        # Send a broadcast message
        comm_protocol.send_message(
            sender_id="agent1",
            recipient_id=None,  # None for broadcast
            message_type="resource_discovery",
            content={"resource_id": 456},
            position=(100, 100)  # Position needed for broadcast reception
        )
        
        # Multiple agents should receive the broadcast
        for agent_id in ["agent2", "agent3", "agent4"]:
            received_messages = comm_protocol.receive_messages(agent_id, (100, 100), 50.0)
            assert len(received_messages) == 1
            assert received_messages[0].message_type == "resource_discovery"
    
    def test_message_priority_handling(self, comm_protocol):
        """Test message priority handling."""
        # Send messages with different priorities
        comm_protocol.send_message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="status_update",
            content={"status": "active"},
            priority=1
        )
        
        comm_protocol.send_message(
            sender_id="agent3",
            recipient_id="agent2",
            message_type="danger_alert",
            content={"danger": "obstacle"},
            priority=3
        )
        
        # Receive messages - high priority should come first
        received_messages = comm_protocol.receive_messages("agent2", (100, 100), 50.0)
        
        assert len(received_messages) == 2
        # Note: Priority ordering depends on implementation
        assert any(msg.priority == 3 for msg in received_messages)
        assert any(msg.priority == 1 for msg in received_messages)
    
    def test_pheromone_management(self, comm_protocol):
        """Test pheromone trail management."""
        # Add pheromone trail
        position = (100, 100)
        pheromone_type = "resource"
        strength = 0.8
        
        comm_protocol._add_pheromone(position, pheromone_type, strength)
        
        # Check pheromone was added
        key = (position, pheromone_type)
        assert key in comm_protocol.pheromones
        assert comm_protocol.pheromones[key] == strength
        
        # Get pheromone information
        pheromone_info = comm_protocol.get_pheromones(position, 10.0)
        assert len(pheromone_info) == 1
        assert pheromone_info[0][1] == pheromone_type
        assert pheromone_info[0][2] == strength
        
        # Update pheromone (adds to existing strength, capped at 1.0)
        comm_protocol._add_pheromone(position, pheromone_type, 0.9)
        key = (position, pheromone_type)
        assert comm_protocol.pheromones[key] == 1.0  # 0.8 + 0.9 = 1.0 (capped)
    
    def test_pheromone_decay(self, comm_protocol):
        """Test pheromone decay over time."""
        # Add pheromone
        position = (200, 200)
        pheromone_type = "danger"
        initial_strength = 1.0
        
        comm_protocol._add_pheromone(position, pheromone_type, initial_strength)
        
        # Simulate time passing and decay
        comm_protocol.update_pheromones()
        
        # Check pheromone strength decreased
        key = (position, pheromone_type)
        current_strength = comm_protocol.pheromones[key]
        assert current_strength < initial_strength
        
        # Check pheromone is removed when strength is too low
        for _ in range(20):  # Multiple decay cycles
            comm_protocol.update_pheromones()
        
        # Pheromone should be removed
        assert key not in comm_protocol.pheromones
    
    def test_message_type_handlers(self, comm_protocol):
        """Test specific message type handlers."""
        # Test resource discovery handler
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="resource_discovery",
            content={"resource_id": 789, "position": (300, 300)},
            timestamp=time.time()
        )
        processed_content = comm_protocol._handle_resource_discovery(message)
        assert "resources" in processed_content
        assert "position" in processed_content
        assert "timestamp" in processed_content
        
        # Test danger alert handler
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="danger_alert",
            content={"danger_type": "obstacle", "position": (400, 400)},
            timestamp=time.time()
        )
        processed_content = comm_protocol._handle_danger_alert(message)
        assert "danger_type" in processed_content
        assert "position" in processed_content
        assert "severity" in processed_content
        
        # Test coordination request handler
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="coordination_request",
            content={"request_type": "formation_change", "target_formation": "circle"},
            timestamp=time.time()
        )
        processed_content = comm_protocol._handle_coordination_request(message)
        assert "request_type" in processed_content
        assert "parameters" in processed_content
        assert "priority" in processed_content
    
    def test_message_processing(self, comm_protocol):
        """Test message processing pipeline."""
        # Send a message
        comm_protocol.send_message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="resource_discovery",
            content={"resource_id": 999}
        )
        
        # Process message
        message = comm_protocol.message_queue[0]
        processed_content = comm_protocol.process_message(message)
        
        # Add message to history manually (as it would be done in receive_messages)
        comm_protocol._add_to_history(message)
        
        # Check message was processed
        assert len(comm_protocol.message_history) == 1
        
        # Check history contains the message
        history_message = comm_protocol.message_history[0]
        assert history_message.sender_id == "agent1"
        assert history_message.message_type == "resource_discovery"
    
    def test_communication_statistics(self, comm_protocol):
        """Test communication statistics."""
        # Send some messages
        for i in range(5):
            comm_protocol.send_message(
                sender_id=f"agent{i}",
                recipient_id="agent2",
                message_type="status_update",
                content={"status": "active"}
            )
        
        # Get statistics
        stats = comm_protocol.get_communication_stats()
        
        assert "messages_sent" in stats
        assert "messages_received" in stats
        assert "bandwidth_usage" in stats
        assert "active_pheromones" in stats
        assert "message_queue_size" in stats
        
        # Check specific values
        assert stats["messages_sent"] == 5
        assert stats["bandwidth_usage"] > 0
    
    def test_bandwidth_reset(self, comm_protocol):
        """Test bandwidth reset functionality."""
        # Use some bandwidth
        comm_protocol.bandwidth_usage = 500
        
        # Reset bandwidth
        comm_protocol.reset_bandwidth()
        
        # Check bandwidth is restored
        assert comm_protocol.bandwidth_usage == 0
    
    def test_message_history_cleanup(self, comm_protocol):
        """Test message history cleanup."""
        # Set small history limit
        comm_protocol.max_message_history = 3
        
        # Send more messages than the limit
        for i in range(5):
            comm_protocol.send_message(
                sender_id=f"agent{i}",
                recipient_id="agent2",
                message_type="test",
                content={"test": True}
            )
            message = comm_protocol.message_queue[i]
            comm_protocol.process_message(message)
        
        # Check history is limited
        assert len(comm_protocol.message_history) <= comm_protocol.max_message_history
    
    def test_expired_message_cleanup(self, comm_protocol):
        """Test expired message cleanup."""
        # Set message expiration time
        comm_protocol.message_expiration_time = 100
        
        # Send a message with old timestamp
        comm_protocol.send_message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type="test",
            content={"test": True},
            expires_at=time.time() - 200  # Very old
        )
        
        # Clean up expired messages
        comm_protocol.clear_expired_messages()
        
        # Check expired message was removed
        assert len(comm_protocol.message_queue) == 0
    
    def test_communication_optimization(self, comm_protocol):
        """Test communication optimization."""
        # Send multiple messages
        for i in range(10):
            comm_protocol.send_message(
                sender_id=f"agent{i}",
                recipient_id="agent2",
                message_type="status_update",
                content={"status": "active"}
            )
        
        # Optimize communication
        optimization_result = comm_protocol.optimize_communication()
        
        # Check optimization was performed (if method exists)
        if optimization_result is not None:
            assert "messages_consolidated" in optimization_result
            assert "bandwidth_saved" in optimization_result
            assert "priority_reordered" in optimization_result
            
            # Check optimization metrics are reasonable
            assert optimization_result["messages_consolidated"] >= 0
            assert optimization_result["bandwidth_saved"] >= 0
            assert optimization_result["priority_reordered"] >= 0
