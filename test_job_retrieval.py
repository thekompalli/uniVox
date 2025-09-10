#!/usr/bin/env python3
"""Test job retrieval directly"""

import asyncio
import sys
from src.repositories.job_repository import JobRepository

async def test_job_retrieval():
    """Test retrieving a specific job"""
    try:
        repo = JobRepository()
        await repo.initialize()
        
        job_id = "411be575-67d9-4eaf-94b6-6a619af5a76b"
        print(f"Testing job retrieval for ID: {job_id}")
        
        job_status = await repo.get_job(job_id)
        
        if job_status:
            print("SUCCESS - Job found:")
            print(f"  ID: {job_status.job_id}")
            print(f"  Status: {job_status.status}")
            print(f"  Progress: {job_status.progress}")
            print(f"  Stage: {job_status.current_stage}")
            print(f"  Created: {job_status.created_at}")
            print(f"  Updated: {job_status.updated_at}")
        else:
            print("FAIL - Job not found")
        
        await repo.cleanup()
        
    except Exception as e:
        print(f"ERROR - Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_job_retrieval())