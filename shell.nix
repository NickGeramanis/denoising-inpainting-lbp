{ pkgs ? import <nixpkgs> {} }:
let
  denoisingInpaintingLbp = pkgs.python313Packages.buildPythonPackage {
    pname = "denoising_inpainting_lbp";
    version = "1.0.0";
    src = ./.;
    format = "setuptools";
    propagatedBuildInputs = with pkgs.python313Packages; [
      opencv-python
      numpy
      matplotlib
    ];
  };
in
pkgs.mkShell {
  buildInputs = [
    (pkgs.python313.withPackages (ps: with ps; [
      denoisingInpaintingLbp
      pytest
      coverage
      pylint
      mypy
    ]))
  ];
}
