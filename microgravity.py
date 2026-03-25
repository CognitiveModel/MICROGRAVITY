import typer
import asyncio
import os
import sys
import logging
from typing import Optional
from pathlib import Path

# Add current directory to path to ensure coding_agent is importable
sys.path.append(os.getcwd())

app = typer.Typer(name="microgravity", help="Microgravity - Advanced Swarm Operating System")

@app.command()
def onboard():
    """Configure Microgravity with API keys."""
    from coding_agent.utils import config
    
    typer.echo("--- Microgravity Onboarding ---")
    bot_token = typer.prompt("Enter your Telegram Bot Token")
    gemini_key = typer.prompt("Enter your Gemini API Key")
    
    env_file = Path(".env")
    content = f"""# Microgravity Configuration
TELEGRAM_BOT_TOKEN={bot_token}
GEMINI_API_KEY={gemini_key}
HUD_ENABLED=1
"""
    env_file.write_text(content)
    typer.echo(f"Configuration saved to {env_file.resolve()}")

@app.command()
def gateway(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace path")
):
    """Start the Microgravity Gateway."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    workspace_path = Path(workspace or os.getcwd()).resolve()
    pid_file = workspace_path / "gateway.pid"
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text())
            os.kill(old_pid, 0)
            typer.secho(f"ERROR: Gateway already running with PID {old_pid}", fg=typer.colors.RED)
            raise typer.Exit(1)
        except (OSError, ValueError):
            pid_file.unlink(missing_ok=True)

    pid_file.write_text(str(os.getpid()))
    typer.echo(f"Starting Microgravity Gateway in {workspace_path} (PID: {os.getpid()})...")
    
    from coding_agent.core.agent import IntrospectionAgent
    
    async def run_services():
        typer.echo("Initializing Microgravity Swarm (Introspection + Swarm + Gateway)...")
        # IntrospectionAgent handles its own Bus, Gateway, and Channel initialization
        agent = IntrospectionAgent()
        
        typer.echo("Gateway Online. Listening for Telegram messages...")
        
        try:
            # We keep the main thread alive while the agent's background tasks handle the work
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            typer.echo("Shutdown initiated...")
            
    try:
        asyncio.run(run_services())
    except KeyboardInterrupt:
        typer.echo("\nGateway stopped by user.")

@app.command()
def agent(
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Start interactive mode")
):
    """Run the agent directly from CLI."""
    from coding_agent.core.agent import IntrospectionAgent
    typer.echo("Initializing Microgravity Agent...")
    agent = IntrospectionAgent()
    
    if message:
        # For simplicity in this CLI, we use the agent's run method
        result = agent.run("general", message)
        typer.echo(f"\n[MICROGRAVITY RESPONSE]\n{result}")
    elif interactive:
        typer.echo("Interactive mode starting... (type 'exit' to quit)")
        while True:
            user_input = typer.prompt("You")
            if user_input.lower() in ["exit", "quit"]:
                break
            result = agent.run("general", user_input)
            typer.echo(f"\n[MICROGRAVITY RESPONSE]\n{result}")

if __name__ == "__main__":
    app()
