format:
	black . -l 79
	linecheck . --fix
install:
	pip install -e .
test:
	pytest -m 'not local'
documentation:
	jupyter-book clean docs/book
	jupyter-book build docs/book
changelog:
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 0.0.0 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org OpenSourceEcon --repo CompMeths --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml setup.py
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml
