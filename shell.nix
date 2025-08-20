let
  pkgs = import <nixpkgs> { };
  ps = pkgs.python3Packages;
in
  ps.buildPythonPackage rec {
    name = "gammy";
    doCheck = false;
    src = ./.;
    depsBuildBuild = with ps; [
      ipython
      pytest
    ];
    propagatedBuildInputs = with ps; [
      bayespy
      h5py
      numpy
      scipy
    ];
  }
