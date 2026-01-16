#!/usr/bin/env python3
"""
Verification script for Flask to FastAPI migration.

This script compares data between Flask and FastAPI databases to ensure
migration was successful.
"""

import asyncio
import logging
from typing import Dict, List, Tuple

import asyncpg
from sqlalchemy import create_engine, MetaData, Table, select, func
import click
from rich.console import Console
from rich.table import Table as RichTable
from rich.progress import track

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class MigrationVerifier:
    """Verifies migration from Flask to FastAPI database."""
    
    def __init__(self, flask_db_url: str, fastapi_db_url: str):
        self.flask_db_url = flask_db_url
        self.fastapi_db_url = fastapi_db_url
        self.flask_engine = None
        self.fastapi_conn = None
        self.verification_results = []
    
    async def connect(self):
        """Establish database connections."""
        self.flask_engine = create_engine(self.flask_db_url)
        self.fastapi_conn = await asyncpg.connect(self.fastapi_db_url)
        logger.info("Connected to both databases")
    
    async def disconnect(self):
        """Close database connections."""
        if self.flask_engine:
            self.flask_engine.dispose()
        if self.fastapi_conn:
            await self.fastapi_conn.close()
    
    async def count_rows(self, flask_table: str, fastapi_table: str) -> Tuple[int, int]:
        """Count rows in both tables."""
        # Count Flask rows
        metadata = MetaData()
        table = Table(flask_table, metadata, autoload_with=self.flask_engine)
        
        with self.flask_engine.connect() as conn:
            flask_count = conn.execute(select(func.count()).select_from(table)).scalar()
        
        # Count FastAPI rows
        fastapi_count = await self.fastapi_conn.fetchval(
            f"SELECT COUNT(*) FROM {fastapi_table}"
        )
        
        return flask_count, fastapi_count
    
    async def verify_table(self, flask_table: str, fastapi_table: str,
                          sample_size: int = 10) -> Dict:
        """Verify a single table migration."""
        try:
            flask_count, fastapi_count = await self.count_rows(flask_table, fastapi_table)
            
            # Sample data comparison
            metadata = MetaData()
            table = Table(flask_table, metadata, autoload_with=self.flask_engine)
            
            with self.flask_engine.connect() as conn:
                # Get sample IDs from Flask
                sample_query = select(table.c.id).limit(sample_size)
                sample_ids = [row[0] for row in conn.execute(sample_query)]
            
            # Verify these IDs exist in FastAPI
            if sample_ids:
                placeholders = ', '.join([f"'{id}'" for id in sample_ids])
                fastapi_sample_count = await self.fastapi_conn.fetchval(
                    f"SELECT COUNT(*) FROM {fastapi_table} WHERE id IN ({placeholders})"
                )
            else:
                fastapi_sample_count = 0
            
            # Calculate verification status
            if flask_count == 0:
                status = "EMPTY"
                match_percentage = 100
            elif flask_count == fastapi_count:
                status = "EXACT_MATCH"
                match_percentage = 100
            elif fastapi_count >= flask_count * 0.95:  # 95% threshold
                status = "GOOD_MATCH"
                match_percentage = (fastapi_count / flask_count) * 100
            else:
                status = "MISMATCH"
                match_percentage = (fastapi_count / flask_count) * 100 if flask_count > 0 else 0
            
            result = {
                'flask_table': flask_table,
                'fastapi_table': fastapi_table,
                'flask_count': flask_count,
                'fastapi_count': fastapi_count,
                'match_percentage': match_percentage,
                'status': status,
                'sample_verification': fastapi_sample_count == len(sample_ids) if sample_ids else True
            }
            
            self.verification_results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error verifying table {flask_table}: {e}")
            result = {
                'flask_table': flask_table,
                'fastapi_table': fastapi_table,
                'error': str(e),
                'status': 'ERROR'
            }
            self.verification_results.append(result)
            return result
    
    async def verify_all_tables(self):
        """Verify all tables in the migration."""
        table_mappings = [
            # Community Edition
            ('tasks', 'tasks'),
            ('workflow_definitions', 'workflow_definitions'),
            ('workflow_instances', 'workflow_instances'),
            ('workflow_steps', 'workflow_steps'),
            ('adapters', 'adapters'),
            ('mcp_servers', 'mcp_servers'),
            ('mcp_tools', 'mcp_tools'),
            ('bridge_connections', 'bridge_connections'),
            
            # Business Edition
            ('approval_workflows', 'approval_workflows'),
            ('approval_steps', 'approval_steps'),
            ('approval_requests', 'approval_requests'),
            ('users', 'users'),
            ('roles', 'roles'),
            ('user_roles', 'user_roles'),
            ('policies', 'policies'),
            ('state_entries', 'state_entries'),
            ('validation_rules', 'validation_rules'),
            
            # Enterprise Edition
            ('tenants', 'tenants'),
            ('tenant_users', 'tenant_users'),
            ('federated_instances', 'federated_instances'),
            ('resource_quotas', 'resource_quotas'),
            ('resource_usage', 'resource_usage'),
            ('audit_logs', 'audit_logs'),
        ]
        
        console.print("\n[bold]Verifying Migration...[/bold]\n")
        
        for flask_table, fastapi_table in track(table_mappings, description="Verifying tables"):
            await self.verify_table(flask_table, fastapi_table)
    
    def display_results(self):
        """Display verification results in a formatted table."""
        table = RichTable(title="Migration Verification Results")
        
        table.add_column("Flask Table", style="cyan")
        table.add_column("FastAPI Table", style="cyan")
        table.add_column("Flask Count", justify="right")
        table.add_column("FastAPI Count", justify="right")
        table.add_column("Match %", justify="right")
        table.add_column("Status", justify="center")
        
        total_flask = 0
        total_fastapi = 0
        errors = 0
        
        for result in self.verification_results:
            if result['status'] == 'ERROR':
                table.add_row(
                    result['flask_table'],
                    result['fastapi_table'],
                    "ERROR",
                    "ERROR",
                    "0%",
                    "[red]ERROR[/red]"
                )
                errors += 1
            else:
                total_flask += result['flask_count']
                total_fastapi += result['fastapi_count']
                
                status_color = {
                    'EXACT_MATCH': '[green]EXACT_MATCH[/green]',
                    'GOOD_MATCH': '[yellow]GOOD_MATCH[/yellow]',
                    'MISMATCH': '[red]MISMATCH[/red]',
                    'EMPTY': '[dim]EMPTY[/dim]'
                }.get(result['status'], result['status'])
                
                table.add_row(
                    result['flask_table'],
                    result['fastapi_table'],
                    str(result['flask_count']),
                    str(result['fastapi_count']),
                    f"{result['match_percentage']:.1f}%",
                    status_color
                )
        
        console.print(table)
        
        # Summary
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"Total Flask records: {total_flask:,}")
        console.print(f"Total FastAPI records: {total_fastapi:,}")
        console.print(f"Overall match rate: {(total_fastapi/total_flask*100) if total_flask > 0 else 0:.1f}%")
        console.print(f"Tables with errors: {errors}")
        
        # Recommendations
        console.print("\n[bold]Recommendations:[/bold]")
        
        mismatches = [r for r in self.verification_results if r.get('status') == 'MISMATCH']
        if mismatches:
            console.print("[red]⚠️  Some tables have mismatched counts:[/red]")
            for m in mismatches:
                console.print(f"   - {m['flask_table']}: {m['flask_count']} → {m['fastapi_count']}")
        
        if errors > 0:
            console.print(f"[red]⚠️  {errors} tables had errors during verification[/red]")
        
        if not mismatches and errors == 0:
            console.print("[green]✅ Migration appears successful![/green]")


@click.command()
@click.option('--flask-db', required=True, help='Flask database URL')
@click.option('--fastapi-db', required=True, help='FastAPI database URL')
@click.option('--detailed', is_flag=True, help='Show detailed verification for each record')
def main(flask_db: str, fastapi_db: str, detailed: bool):
    """Verify AICtrlNet migration from Flask to FastAPI."""
    verifier = MigrationVerifier(flask_db, fastapi_db)
    
    async def run_verification():
        await verifier.connect()
        try:
            await verifier.verify_all_tables()
            verifier.display_results()
        finally:
            await verifier.disconnect()
    
    asyncio.run(run_verification())


if __name__ == '__main__':
    # Add rich to requirements for pretty output
    try:
        import rich
    except ImportError:
        console.print("[yellow]Install 'rich' for better output: pip install rich[/yellow]")
    
    main()