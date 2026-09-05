"""Flask entry point for the ParticlePHINGAN playground.

Run from the repo root with the venv active:

    python -m playground.server                   # default 127.0.0.1:5050
    PORT=5060 python -m playground.server         # custom port
    PLAYGROUND_DEBUG=1 python -m playground.server  # debug mode (auto reload)

All physics modules import via the `src/` source root, so launching from the
repo root is mandatory.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

from flask import Flask, render_template

from playground.api.admin import bp as admin_bp
from playground.api.run_context import bp as run_context_bp
from playground.api.g4_data import bp as g4_data_bp
from playground.api.samplers import bp as samplers_bp
from playground.api.pdfs import bp as pdfs_bp
from playground.api.generators import bp as generators_bp
from playground.api.stepping import bp as stepping_bp
from playground.api.pull import bp as pull_bp
from playground.api.pairwise import bp as pairwise_bp
from playground.api.table import bp as table_bp
from playground.api.cluster import bp as cluster_bp


HERE = Path(__file__).parent


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=str(HERE / "static"),
        template_folder=str(HERE / "templates"),
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    app.register_blueprint(admin_bp,       url_prefix="/api/admin")
    app.register_blueprint(run_context_bp, url_prefix="/api/run_context")
    app.register_blueprint(g4_data_bp,    url_prefix="/api/g4")
    app.register_blueprint(samplers_bp,   url_prefix="/api/samplers")
    app.register_blueprint(pdfs_bp,       url_prefix="/api/pdfs")
    app.register_blueprint(generators_bp, url_prefix="/api/generators")
    app.register_blueprint(stepping_bp,   url_prefix="/api/stepping")
    app.register_blueprint(pull_bp,       url_prefix="/api/pull")
    app.register_blueprint(pairwise_bp,   url_prefix="/api/pairwise")
    app.register_blueprint(table_bp,      url_prefix="/api/table")
    app.register_blueprint(cluster_bp,    url_prefix="/api/cluster")

    @app.errorhandler(500)
    def server_error(err):
        logging.exception("server error: %s", err)
        return {"error": str(err)}, 500

    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app()
    port = int(os.environ.get("PORT", "5050"))
    debug = bool(int(os.environ.get("PLAYGROUND_DEBUG", "0")))
    app.run(host="127.0.0.1", port=port, debug=debug, use_reloader=debug)


if __name__ == "__main__":
    main()
