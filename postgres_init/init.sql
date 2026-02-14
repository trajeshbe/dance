-- Initialize PostgreSQL database for Dance Video Generator

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Database is created by docker-compose environment variables
-- This file can contain additional initialization SQL if needed

-- Create initial admin user (optional)
-- INSERT INTO users (id, email, created_at)
-- VALUES (uuid_generate_v4(), 'admin@example.com', NOW());
