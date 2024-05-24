let
  pkgs = import <nixpkgs-unstable> {};
  pythonPkgs = pkgs.python311Packages;
in pkgs.mkShell {

  name = "gammy-shell";

  PYTHONPATH = toString ./.;

  propagatedBuildInputs = with pythonPkgs; [

    # Core packages
    bayespy
    h5py
    numpy
    pandas
    pydantic
    pytest
    scipy

    # Dev tools
    ipython
    matplotlib
    pip
    plotly

    # Code quality
    autopep8
    pyflakes
    pylint

    pkgs.nodePackages.pyright

  ];
}
