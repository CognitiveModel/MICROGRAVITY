# Microgravity Gateway

Microgravity is an advanced swarm-based AI operating system designed for autonomous task execution and UI automation. It rebranded from "Fortress" to provide a more streamlined, secure, and powerful experience for agent-driven workflows.

## Features
- **Swarm Intelligence**: Multi-agent coordination for complex objectives.
- **UI Automation**: Autonomous control over desktop applications and web browsers.
- **Secure Configuration**: Environment-based secret management (using `.env`).
- **Telegram Integration**: Remote command and control via Telegram.
- **Experiential Learning**: Agents learn from past successes and failures to optimize future tasks.

## Getting Started

### Prerequisites
- Python 3.10+
- Telegram Bot Token (from @BotFather)
- Gemini API Key (from Google AI Studio)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shreychhedareal/MICROGRAVITY.git
   cd MICROGRAVITY
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your API keys:
   ```bash
   python microgravity.py onboard
   ```
   *This will create a local `.env` file. Do not share this file.*

### Running the Gateway
Start the Microgravity Gateway with:
```bash
microgravity gateway
```

## Roadmap
See [FEATURE_MANIFEST.md](FEATURE_MANIFEST.md) for technical details on the learning architecture.
Future plans include Cloud Database integration and enhanced HUD UI.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[MIT License](LICENSE) (or specify your preferred license)

---
**Maintained by**: [Aryan Malik](mailto:aryanmalik77g@gmail.com)
