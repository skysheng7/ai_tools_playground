# AI Tools Playground - Copilot Instructions

## Project Overview
This is a Jupyter Book project focused on AI tools and prompt engineering for data science. The book is automatically published to GitHub Pages via workflow automation and covers comprehensive AI development topics from fundamentals to advanced implementations.

## Architecture & Structure

### Core Components
- **Book Engine**: Uses Jupyter Book (`jupyter-book` package) for content generation and HTML publishing
- **Content Structure**: Organized as a structured educational book with chapters covering AI fundamentals, tools, applications, and advanced topics
- **Deployment**: Automated GitHub Pages deployment via `.github/workflows/deploy.yml`
- **Content Format**: Mix of Markdown files and potential Jupyter notebooks (`.md` files in chapters)

### Key Configuration Files
- `_config.yml`: Central book configuration (title, author, repository info, extensions)
- `_toc.yml`: Table of contents defining book structure (currently minimal - needs expansion)
- `requirements.txt`: Python dependencies for book building and content execution
- `.github/workflows/deploy.yml`: Automated deployment pipeline

## Essential Workflows

### Content Development
```bash
# Build book locally for testing
jupyter-book build .

# View built content
open _build/html/index.html
```

### Adding New Chapters
1. Create new `.md` file (can be in root or `chapters/` subdirectory)
2. Add entry to `_toc.yml` following existing format (NO file extensions in TOC)
3. Build locally to verify before pushing

### Managing Dependencies
- **Critical**: Always update `requirements.txt` when using new Python packages in content
- Deployment will fail if packages are missing from requirements.txt
- Current packages include: jupyter-book, pandas, matplotlib, numpy, scikit-learn, torch, transformers, seaborn, plotly, ipywidgets, jupyterlab

## Project-Specific Patterns

### Chapter Organization
Based on built content, chapters follow this structure:
- **Fundamentals**: introduction-to-ai, llm-basics, prompt-engineering
- **AI Tools & Applications**: ai-development-tools, practical-applications, case-studies  
- **Advanced Topics**: fine-tuning, evaluation-metrics, ethical-considerations

### TOC Structure Inconsistency
⚠️ **Current Issue**: `_toc.yml` only contains root page but built site shows full chapter structure. This suggests chapters may be auto-discovered or there's a missing TOC configuration that needs updating.

### Content Standards
- Chapters include practical code examples with proper syntax highlighting
- Real-world case studies with specific metrics (e.g., "35% increase in click-through rates")
- Comprehensive coverage from basic concepts to implementation details
- Focus on UBC-approved vs restricted AI tools (see intro.md for policy details)

## Development Guidelines

### Local Development
1. Ensure `jupyter-book` is installed: `pip install jupyter-book`
2. Build with: `jupyter-book build .`
3. Test changes locally before committing
4. Check for build errors in terminal output

### Content Creation
- Use MyST Markdown syntax for enhanced features (enabled extensions in `_config.yml`)
- Include practical code examples and real-world applications
- Reference specific tools and frameworks mentioned in existing chapters
- Maintain educational progression from fundamentals to advanced topics

### GitHub Integration
- Repository: `skysheng7/ai_tools_playground`
- Auto-deployment on pushes to `main` branch
- GitHub Pages serves content from `_build/html`
- Issues and repository buttons enabled in book interface

## Critical Integration Points

### Existing Cursor Rules
Located in `.cursor/rules/website-launch.mdc` - contains comprehensive Jupyter Book setup instructions and workflow guidance that should be referenced for consistency.

### External Dependencies
- GitHub Actions for deployment (Ubuntu 20.04, Python 3.11)
- GitHub Pages for hosting
- Python ML ecosystem packages (PyTorch, Transformers, etc.)

## Common Tasks

### Adding Content
- Create `.md` files with proper MyST syntax
- Include in `_toc.yml` (without file extensions)
- Update `requirements.txt` if using new packages
- Test locally with `jupyter-book build .`

### Troubleshooting Builds
- Check `requirements.txt` for missing dependencies
- Verify TOC syntax (no file extensions, proper indentation)
- Run `jupyter-book build . -v` for verbose error output
- Ensure all referenced files exist and paths are correct
