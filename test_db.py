#!/usr/bin/env python3
"""Test database connectivity and job storage"""

import asyncio
import asyncpg
from datetime import datetime
from src.config.database_config import DatabaseConfig

async def test_db():
    """Test database connectivity"""
    try:
        db_config = DatabaseConfig()
        
        # Create connection
        pool = await asyncpg.create_pool(
            database=db_config.postgres_db,
            user=db_config.postgres_user,
            password=db_config.postgres_password,
            host=db_config.postgres_host,
            port=db_config.postgres_port,
            min_size=1,
            max_size=5
        )
        
        print("OK Database connection successful")
        
        # Check if jobs table exists
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'jobs'"
            )
            print(f"OK Jobs table exists: {result > 0}")
            
            if result > 0:
                # Check job count
                job_count = await conn.fetchval("SELECT COUNT(*) FROM jobs")
                print(f"OK Jobs in database: {job_count}")
                
                if job_count > 0:
                    # Show jobs
                    jobs = await conn.fetch("SELECT job_id, state, created_at FROM jobs ORDER BY created_at DESC LIMIT 5")
                    print("OK Recent jobs:")
                    for job in jobs:
                        print(f"  - {job['job_id']}: {job['state']} (created: {job['created_at']})")
        
        await pool.close()
        
    except Exception as e:
        print(f"ERROR Database test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_db())