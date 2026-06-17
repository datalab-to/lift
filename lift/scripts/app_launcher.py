import os
import sys


def main():
    """Launch the Schema Studio Streamlit app via the `lift_app` console script."""
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        raise ImportError(
            "Schema Studio requires the app dependencies. "
            "Install with: pip install lift-pdf[app]"
        )

    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    sys.argv = ["streamlit", "run", app_path, *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
