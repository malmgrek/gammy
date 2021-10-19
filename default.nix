let
  pkgs = import <nixpkgs> { };
  ps = pkgs.python3Packages.override {
    overrides = self: super: {
      # Nixpkgs uses jedi version 0.17.2 which
      # breaks Emacs anaconda-mode-complete.
      jedi = super.jedi.overrideAttrs (oldAttrs: {
        src = super.fetchPypi {
          pname = "jedi";
          version = "0.18.0";
          sha256 = "01q7xla9ccjra3j4nhb1lvn4kv8z8sdfqdx1h7cgx2md9d00lmcj";
        };
      });
    };
  };
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
