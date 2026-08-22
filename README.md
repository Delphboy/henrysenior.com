# henrysenior.com

The source code for [henrysenior.com](https://henrysenior.com/) — the personal website of
[Henry Senior](https://henrysenior.com/), Lecturer in Computer Science at the University of Salford.

The site is built with [Jekyll](https://jekyllrb.com/) using the
[Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) theme (via the
[academicpages](https://academicpages.github.io/) fork) and is deployed with
[Netlify](https://www.netlify.com/) (see [`netlify.toml`](netlify.toml)).

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Setup (one-time)](#setup-one-time)
- [Run the development server](#run-the-development-server)
- [Build the site for production](#build-the-site-for-production)
- [Deployment](#deployment)
- [Project structure](#project-structure)
- [Adding content](#adding-content)
- [Troubleshooting](#troubleshooting)

## Prerequisites

You need Ruby and Bundler:

| Tool    | Minimum version | Check with        |
|---------|-----------------|-------------------|
| Ruby    | 2.7 (3.x works) | `ruby --version`  |
| Bundler | 2.x             | `bundle --version` |

> **Windows only:** Ruby is easiest to install via the
> [RubyInstaller](https://rubyinstaller.org/) (choose the "Devkit" variant). The
> Chocolatey route also works:
>
> ```powershell
> choco install ruby
> choco install msys2
> ridk install
> ```

## Setup (one-time)

Clone the repo and install the Ruby dependencies:

```bash
git clone git@github.com:delphboy/henrysenior.com.git
cd henrysenior.com
bundle install
```

`bundle install` reads the `Gemfile` and creates a `Gemfile.lock` pinning every
dependency, so the site builds identically on all machines.

If you ever change the `Gemfile`, run `bundle install` again to update the lock
file, and `bundle update <gem-name>` to upgrade a specific gem.

## Run the development server

Start a local server with live-reload and full error traces:

```bash
bundle exec jekyll serve --config _config.yml,_config.dev.yml --livereload --trace
```

Then open <http://localhost:4000> in your browser.

- The `--config _config.yml,_config.dev.yml` part matters: `_config.yml` sets
  `url: https://henrysenior.com`, so without the dev overrides every internal
  link (the navigation bar included) points at the live site instead of your
  local build. `_config.dev.yml` overrides `url` so links resolve to
  `http://localhost:4000/...` (or root-relative like `/about/` when building
  with `jekyll build`). Production deploys don't use this file.
- Any change to a page, post, or layout is rebuilt automatically (Jekyll watches
  the source directory).
- `--livereload` pushes changes to the browser without a manual refresh.
- `--trace` prints the full backtrace if the build fails, which makes errors
  much easier to debug.
- To also preview unpublished drafts, add `--drafts`.

> Note: changes to `_config.yml` (or `_config.dev.yml`) require a restart of
> the server — Jekyll does not reload them automatically.

## Build the site for production

To generate the static site into `_site/` without running a server:

```bash
bundle exec jekyll build
```

Set `JEKYLL_ENV=production` to match the production build (Netlify does this for
you):

```bash
JEKYLL_ENV=production bundle exec jekyll build
```

The contents of `_site/` are exactly what gets deployed — upload that directory
to any static host if you aren't using Netlify/GitHub Pages.

## Deployment

### Netlify (current)

`netlify.toml` is already configured — pushes to `master` auto-deploy:

```toml
[build]
  command = "jekyll build"
  publish = "_site"

[build.environment]
  JEKYLL_ENV = "production"
```

### GitHub Pages (alternative)

This is a plain Jekyll project, so it can be served from GitHub Pages too. Push
the `master` branch and enable Pages in the repo settings, or publish the
`_site/` directory to a `gh-pages` branch.

## Project structure

```
.
├── _config.yml          # Site-wide configuration (title, author, plugins, defaults)
├── Gemfile              # Ruby dependencies (Jekyll, plugins)
├── netlify.toml         # Netlify build configuration
├── _data/               # YAML data files (navigation.yml, authors.yml, ui-text.yml)
├── _layouts/            # HTML templates (single, archive, talk, ...)
├── _includes/           # Reusable HTML snippets (author-profile, archive-single, ...)
├── _sass/               # SCSS stylesheets
├── assets/              # Static assets (JS, CSS)
├── images/              # Site images
├── files/               # Paper PDFs
├── _pages/              # Top-level pages (index, publications, projects, demos, words, ...)
├── _posts/              # Blog posts (the "Words" page), named YYYY-MM-DD-slug.md
├── _publications/       # Publication entries
├── _projects/           # Project entries
├── _demos/              # Interactive demo entries
├── _fyp/                # Final Year Project group content
└── _site/               # Generated output (do not edit; gitignored)
```

## Adding content

Most content is plain Markdown files with YAML front matter, organised into
[collections](https://jekyllrb.com/docs/collections/) defined in `_config.yml`.

### Blog post (`_posts/`)

```markdown
---
title: "My New Post"
date: 2026-02-01
---

Post body in Markdown.
```

### Publication (`_publications/`)

```markdown
---
title: "Paper title"
collection: publications
date: 2026-02-01
venue: 'Conference or journal name'
excerpt: "One-line summary shown on the publications page."
paperurl: '/files/publications/my-paper.pdf'
link: 'https://doi.org/...'
github: https://github.com/yourname/repo
citation: '<b>Author</b> (2026). Paper title. <i>Venue</i>'
---

Full abstract and details.
```

The archive page shows the venue, date, and icons for the link/PDF/PDF code
links automatically. See the existing files in `_publications/` for examples.

### Projects (`_projects/`), demos (`_demos/`), FYP (`_fyp/`)

The same pattern applies — see the existing files for the exact front matter
fields each collection expects. After adding a file, the entry appears on its
archive page automatically.

## Troubleshooting

**`bundler: command not found`**
Bundler isn't installed — run `gem install bundler`.

**`cannot load such file -- webrick`**
Ruby 3.0+ no longer ships the `webrick` gem, which Jekyll's dev server needs.
`webrick` is already declared in the `Gemfile`, so `bundle install` fixes this.

**`Failed to build gem native extension`**
A gem needs a compiler. On macOS install Xcode Command Line Tools
(`xcode-select --install`); on Debian/Ubuntu install `build-essential` and
`ruby-dev`; on Windows make sure the RubyInstaller Devkit is installed.

**`Liquid Warning: Liquid syntax error`**
Liquid inside `{{ }}` is parsed even inside HTML comments. Use Jekyll's
`{% comment %}...{% endcomment %}` tags instead of `<!-- ... -->` when you want
to comment out template code.

**Port 4000 already in use**
Run the server on a different port: `bundle exec jekyll serve -P 4001`.

**Changes to `_config.yml` have no effect**
Restart the server — Jekyll only reads the config at startup.
