# Development Guidelines for Autonomous Agents

## Build/Lint/Test Commands

### Code Quality
```bash
ruff check .          # Lint all Python files
ruff format .         # Format all Python files
```

### Dependency Management
```bash
# Install dependencies (preferred method)
uv sync
```

### Type Hints
- All functions should use type hints
- Use TypedDict or dict for agent/subagent configuration

### Console Output with Rich
- Use `console.print()` with rich markup: `[color]text[/color]`
- Use `Panel.fit()` for important messages
- Use `Tree()` for hierarchical data display
- Use `Table()` for tabular data
- Icons: 🎯 🧠 🗃️ 🐍 📊 📝 ✅ ❌
