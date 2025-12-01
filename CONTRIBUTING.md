# Contributing to AI Space Exploration

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🌟 Ways to Contribute

- 🐛 Report bugs and issues
- 💡 Suggest new features or enhancements
- 📝 Improve documentation
- 🧪 Write tests
- 💻 Submit code improvements
- 🎨 Improve UI/UX
- 📊 Add datasets or models

## 🚀 Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/karimosman89/AI-for-Space-Exploration-and-Autonomous-Astronomy.git
cd AI-for-Space-Exploration-and-Autonomous-Astronomy
```

### 2. Set Up Development Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## 💻 Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting: `black src/`
- Use type hints where appropriate
- Write docstrings for all public functions and classes

### Testing

- Write unit tests for new features
- Ensure all tests pass: `pytest tests/`
- Aim for >80% code coverage

### Documentation

- Update README.md if needed
- Add docstrings to new functions
- Create examples for new features
- Update API documentation

## 📋 Pull Request Process

### Before Submitting

1. **Test your changes**:
   ```bash
   pytest tests/ -v
   black src/
   flake8 src/
   ```

2. **Update documentation**:
   - Add docstrings
   - Update README if needed
   - Add examples

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

### Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### PR Guidelines

1. **Create Pull Request**:
   - Provide clear description
   - Reference related issues
   - Include screenshots/demos if applicable

2. **PR Checklist**:
   - [ ] Code follows project style guidelines
   - [ ] Tests pass locally
   - [ ] Documentation updated
   - [ ] No merge conflicts
   - [ ] Changes are focused and atomic

3. **Review Process**:
   - Maintainers will review within 3-5 days
   - Address feedback and requested changes
   - Keep PR updated with main branch

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
**Describe the bug**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. ...
2. ...

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10]
- Package version: [e.g., 1.0.0]

**Additional context**
Any other relevant information.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions or features you've considered.

**Additional context**
Any other relevant information or screenshots.
```

## 🎯 Priority Areas

We're especially interested in contributions in these areas:

1. **New Models**:
   - Advanced deep learning architectures
   - Transfer learning implementations
   - Model optimization

2. **Data Processing**:
   - Data augmentation techniques
   - New dataset integrations
   - Preprocessing pipelines

3. **Deployment**:
   - Cloud deployment scripts
   - Kubernetes configurations
   - Edge deployment solutions

4. **Documentation**:
   - Tutorials and guides
   - API documentation
   - Example notebooks

5. **Testing**:
   - Unit tests
   - Integration tests
   - Performance benchmarks

## 📞 Communication

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: karim.osman@example.com for private matters

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for everyone.

### Our Standards

- ✅ Be respectful and inclusive
- ✅ Welcome newcomers
- ✅ Give constructive feedback
- ✅ Focus on what's best for the community
- ❌ No harassment or discrimination
- ❌ No trolling or insulting comments
- ❌ No publishing private information

### Enforcement

Violations can be reported to karim.osman@example.com. All complaints will be reviewed and investigated promptly and fairly.

## 📚 Additional Resources

- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Python Style Guide](https://pep8.org/)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)

## ❓ Questions?

Don't hesitate to ask! Open an issue with the "question" label or start a discussion.

---

**Thank you for contributing! 🚀**

Your efforts help make space exploration accessible to everyone.
