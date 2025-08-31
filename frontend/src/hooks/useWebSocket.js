import { useState, useEffect, useRef, useCallback } from 'react';
import { toast } from 'react-toastify';

const useWebSocket = (url, options = {}) => {
  const [socket, setSocket] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [readyState, setReadyState] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState('Connecting');
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = options.maxReconnectAttempts || 5;
  const reconnectInterval = options.reconnectInterval || 3000;

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        setReadyState(1);
        setConnectionStatus('Connected');
        reconnectAttempts.current = 0;
        toast.success('WebSocket connected successfully');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        setReadyState(3);
        setConnectionStatus('Disconnected');
        
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current += 1;
          setConnectionStatus(`Reconnecting... (${reconnectAttempts.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else {
          toast.error('WebSocket connection failed after maximum attempts');
          setConnectionStatus('Failed');
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        toast.error('WebSocket connection error');
      };

      setSocket(ws);
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      toast.error('Failed to create WebSocket connection');
    }
  }, [url, maxReconnectAttempts, reconnectInterval]);

  const sendMessage = useCallback((message) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      toast.error('WebSocket is not connected');
    }
  }, [socket]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (socket) {
      socket.close();
    }
  }, [socket]);

  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    socket,
    lastMessage,
    readyState,
    connectionStatus,
    sendMessage,
    disconnect,
    reconnect: connect
  };
};

export default useWebSocket;