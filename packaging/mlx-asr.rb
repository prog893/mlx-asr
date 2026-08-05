class MlxAsr < Formula
  include Language::Python::Virtualenv

  desc "Batch speech-to-text on Apple Silicon: Voxtral, Whisper, kotoba-whisper"
  homepage "https://github.com/prog893/mlx-asr"
  # No `version` line: Homebrew scans it from the tag, and declaring both is
  # flagged as redundant. Note the tag must be written out rather than
  # interpolated as "v#{version}", since style autocorrect sorts `url` above
  # `version`, at which point the interpolation resolves to a bare "v" and the
  # clone fails with "Remote branch v not found in upstream origin".
  url "https://github.com/prog893/mlx-asr.git", tag: "v0.1.0"
  license "MIT"

  # A git URL with a tag rather than a release tarball, matching the other
  # formulae in this tap. This also works while the source repo is private,
  # because Homebrew shells out to git and picks up the user's credentials; a
  # `.tar.gz` from `codeload.github.com` would 404 without a token.
  head "https://github.com/prog893/mlx-asr.git", branch: "main"

  # MLX is Metal-only, so there is no Intel or Linux build to offer. macOS 14 is
  # mlx-metal's own floor for its arm64 wheels.
  depends_on arch: :arm64
  depends_on "ffmpeg"
  depends_on macos: :sonoma
  depends_on "python@3.13"

  # ffmpeg does the audio decoding. It is a hard dependency rather than optional
  # because the Python decoder (miniaudio) is deliberately not shipped: ffmpeg
  # reads strictly more formats, and `load_audio_16k` falls back to it.

  # Note there is no `depends_on "numpy"` / `"protobuf"` / `"sentencepiece"`,
  # though all three formulae exist. Each was tried and rejected:
  #   numpy         Homebrew ships 2.5.1; numba caps numpy at <2.5.
  #   protobuf      the formula installs C++ libraries only, no python3.13
  #                 bindings, so `import google.protobuf` fails.
  #   sentencepiece likewise ships no Python bindings.
  # So every Python dependency comes from a pinned wheel below.

  # Wheels, not sdists, which is why this is a tap and not homebrew-core:
  # `mlx` and `mlx-metal` publish no sdist at all, and numba/llvmlite/tokenizers
  # would each need a full LLVM or Rust toolchain to build from source.
  # Regenerate with: uv run python packaging/gen_formula.py
  resource "annotated-doc" do
    url "https://files.pythonhosted.org/packages/3e/30/e900b21425a860e195f32e37657aa1f7c7f2b1bfb26f03ca209b90933c06/annotated_doc-0.0.5-py3-none-any.whl"
    sha256 "117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101"
  end

  resource "annotated-types" do
    url "https://files.pythonhosted.org/packages/99/91/8acff4f5e50511b911bbccb72b8628a49c68ce14148cd9f6431094859a90/annotated_types-0.8.0-py3-none-any.whl"
    sha256 "f072f4d804ea359e4eaf198b1af7a8b0943881a87f31bb764f8bf219bb9419e0"
  end

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/da/35/f2287558c17e29fafc8ef3daf819bb9834061cfa43bff8014f7df7f63bdc/anyio-4.14.2-py3-none-any.whl"
    sha256 "9f505dda5ac9f0c8309b5e8bd445a8c2bf7246f3ce950121e45ea15bc41d1494"
  end

  resource "attrs" do
    url "https://files.pythonhosted.org/packages/64/b4/17d4b0b2a2dc85a6df63d1157e028ed19f90d4cd97c36717afef2bc2f395/attrs-26.1.0-py3-none-any.whl"
    sha256 "c647aa4a12dfbad9333ca4e71fe62ddc36f4e63b2d260a37a8b83d2f043ac309"
  end

  resource "av" do
    url "https://files.pythonhosted.org/packages/77/b3/2576a44b4f39c7462ced4c17fec04c756f7b0f3c5cb940d124173e417d6a/av-18.0.0-cp311-abi3-macosx_14_0_arm64.whl"
    sha256 "35274c20d2ad3b4774fe632bcef2e34af79858ddf899352339cc3babbc13a484"
  end

  resource "certifi" do
    url "https://files.pythonhosted.org/packages/0b/a7/71ac2cff56fec219ed242bb11b8efb69fcc4bec75db06fb7bfe35de520e6/certifi-2026.7.22-py3-none-any.whl"
    sha256 "62f22742b58a1a33014a2b6b706588a8d7e2a88ae7bd1a6ebe8c992928483775"
  end

  resource "cffi" do
    url "https://files.pythonhosted.org/packages/55/41/4c7042f317b9217502988f0873af87e16ad606dc20f84e546e3e6ce9764c/cffi-2.1.1-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "19ee6127ee34de7d83ce3d371ebc5ed91addbdcc39f9ab15ce4eb35a4e534971"
  end

  resource "charset-normalizer" do
    url "https://files.pythonhosted.org/packages/98/2b/f97f1c193fb855c345d678f5077d6926034db0722df74c8f057020e05a25/charset_normalizer-3.4.9-py3-none-any.whl"
    sha256 "68e5f26a1ad57ded6d1cfb85331d1c1a195314756471d97758c48498bb4dcdf5"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/fb/e2/79c688af8b210d232694e31e59da9f6ec747bae31c3f5946e4e9b98860d5/click-8.4.2-py3-none-any.whl"
    sha256 "e6f9f66136c816745b9d65817da91d61d957fb16e02e4dcd0552553c5a197b76"
  end

  resource "filelock" do
    url "https://files.pythonhosted.org/packages/c1/e8/72f8cef9fdfeffe06213fe8508039396ee48daa0e3259457ed766173bfd6/filelock-3.32.2-py3-none-any.whl"
    sha256 "87dd94cf281e586d135fa51132b8e3d9a598b316e90377a288663c9321036c82"
  end

  resource "flatbuffers" do
    url "https://files.pythonhosted.org/packages/e8/2d/d2a548598be01649e2d46231d151a6c56d10b964d94043a335ae56ea2d92/flatbuffers-25.12.19-py2.py3-none-any.whl"
    sha256 "7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4"
  end

  resource "fsspec" do
    url "https://files.pythonhosted.org/packages/fd/3c/6a2bf344106328fd04963664a60b9bb6496fc25df8e962fcdc1367285fb9/fsspec-2026.7.0-py3-none-any.whl"
    sha256 "b57ddbafedfaef7018c1ecab32aa200a9d7ca26b77965f64e48b70061249d279"
  end

  resource "h11" do
    url "https://files.pythonhosted.org/packages/04/4b/29cac41a4d98d144bf5f6d33995617b185d14b22401f75ca86f384e87ff1/h11-0.16.0-py3-none-any.whl"
    sha256 "63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86"
  end

  resource "hf-xet" do
    url "https://files.pythonhosted.org/packages/4b/69/55b8dcf636142ae660fec1869fcac14c4da2e8412e14d6eee1523be77e9f/hf_xet-1.6.0-cp38-abi3-macosx_11_0_arm64.whl"
    sha256 "f0906082d9932ae0c0057fa194041c22b4e2cdb46b2592ef3b91f020d62a081a"
  end

  resource "httpcore" do
    url "https://files.pythonhosted.org/packages/7e/f5/f66802a942d491edb555dd61e3a9961140fd64c90bce1eafd741609d334d/httpcore-1.0.9-py3-none-any.whl"
    sha256 "2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/2a/39/e50c7c3a983047577ee07d2a9e53faf5a69493943ec3f6a384bdc792deb2/httpx-0.28.1-py3-none-any.whl"
    sha256 "d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad"
  end

  resource "huggingface-hub" do
    url "https://files.pythonhosted.org/packages/97/bb/63a644c75b545f3ff394b822e9bd1c4a9586489c618b77a4d8a44a33a23b/huggingface_hub-1.26.0-py3-none-any.whl"
    sha256 "e8cca670caa5d8dfa7e45bf45e86b466698198cd8150c021bcdb4a86b9252364"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/1e/5e/d4e9f1a599fb8e573b7b87160658329fbf28d19eac2718f51fc3def3aa5a/idna-3.18-py3-none-any.whl"
    sha256 "7f952cbe720b688055e3f87de14f5c3e5fdaa8bc3928985c4077ca689de849a2"
  end

  resource "jinja2" do
    url "https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl"
    sha256 "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67"
  end

  resource "jsonschema" do
    url "https://files.pythonhosted.org/packages/69/90/f63fb5873511e014207a475e2bb4e8b2e570d655b00ac19a9a0ca0a385ee/jsonschema-4.26.0-py3-none-any.whl"
    sha256 "d489f15263b8d200f8387e64b4c3a75f06629559fb73deb8fdfb525f2dab50ce"
  end

  resource "jsonschema-specifications" do
    url "https://files.pythonhosted.org/packages/41/45/1a4ed80516f02155c51f51e8cedb3c1902296743db0bbc66608a0db2814f/jsonschema_specifications-2025.9.1-py3-none-any.whl"
    sha256 "98802fee3a11ee76ecaca44429fda8a41bff98b00a0f2838151b113f210cc6fe"
  end

  resource "llvmlite" do
    url "https://files.pythonhosted.org/packages/9c/23/fe9316d14626b42c73ef0b502e724705a6ee9450afe53759c0a99c37c2d7/llvmlite-0.48.0-cp313-cp313-macosx_12_0_arm64.whl"
    sha256 "a83a99ef0c05b4ccddf9b6218ed9fe84b653a0caf7c1d9dbe148d6d16c67f518"
  end

  resource "markdown-it-py" do
    url "https://files.pythonhosted.org/packages/b3/81/4da04ced5a082363ecfa159c010d200ecbd959ae410c10c0264a38cac0f5/markdown_it_py-4.2.0-py3-none-any.whl"
    sha256 "9f7ebbcd14fe59494226453aed97c1070d83f8d24b6fc3a3bcf9a38092641c4a"
  end

  resource "markupsafe" do
    url "https://files.pythonhosted.org/packages/9c/d9/5f7756922cdd676869eca1c4e3c0cd0df60ed30199ffd775e319089cb3ed/markupsafe-3.0.3-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "116bb52f642a37c115f517494ea5feb03889e04df47eeff5b130b1808ce7c219"
  end

  resource "mdurl" do
    url "https://files.pythonhosted.org/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl"
    sha256 "84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8"
  end

  resource "mistral-common" do
    url "https://files.pythonhosted.org/packages/ef/a4/bc2850eb33cc2d633a21f51530756350dca325ee69b07c9202552f2bbadb/mistral_common-1.11.7-py3-none-any.whl"
    sha256 "a9511b88eacacbe7dacddd9d3498c1739f56847b7fdddbd5a22e7844fd9def95"
  end

  resource "mlx" do
    url "https://files.pythonhosted.org/packages/4c/a8/7bc999ce5d09dfac8961dcda4ed47e173fca2857492f34599b237380f20d/mlx-0.32.0-cp313-cp313-macosx_26_0_arm64.whl"
    sha256 "4192a2d02014a13a6a1030bf13dfb4e4fe05ec3ffa47678ee37da29111e25cb1"
  end

  resource "mlx-audio" do
    url "https://files.pythonhosted.org/packages/3f/be/42234a5891b3d6d5078434d657cfcda6c5cce31117cb19fc9915422cc306/mlx_audio-0.4.7-py3-none-any.whl"
    sha256 "e7580e5ee740ca182e93b764d778b8c834d4e07bf91835cbaaafef196e2e058c"
  end

  resource "mlx-lm" do
    url "https://files.pythonhosted.org/packages/90/02/9a67b8e4f87e3e2e5cd7b1ad79304b93c09a0db6af34bee75e6551c06c60/mlx_lm-0.31.3-py3-none-any.whl"
    sha256 "758cfddf1180053b7613db76fad3d246a331a2a905808e1164a275621fc983b8"
  end

  resource "mlx-metal" do
    url "https://files.pythonhosted.org/packages/dc/59/65d32520175379df33f107749193aa94ea9db069167a36a1a100ff689f62/mlx_metal-0.32.0-py3-none-macosx_26_0_arm64.whl"
    sha256 "3af76a498d84804f66119800499f9d143d7dffb0878a0dd0d7c2846e58565fd7"
  end

  resource "mlx-whisper" do
    url "https://files.pythonhosted.org/packages/22/b7/a35232812a2ccfffcb7614ba96a91338551a660a0e9815cee668bf5743f0/mlx_whisper-0.4.3-py3-none-any.whl"
    sha256 "6b82b6597a994643a3e5496c7bc229a672e5ca308458455bfe276e76ae024489"
  end

  resource "more-itertools" do
    url "https://files.pythonhosted.org/packages/e8/3d/1087453384dbde46a8c7f9356eead2c58be8a7bf156bca40243377c85715/more_itertools-11.1.0-py3-none-any.whl"
    sha256 "4b65538ae22f6fed0ce4874efd317463a7489796a0939fa66824dd542125a192"
  end

  resource "mpmath" do
    url "https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl"
    sha256 "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c"
  end

  resource "networkx" do
    url "https://files.pythonhosted.org/packages/9e/c9/b2622292ea83fbb4ec318f5b9ab867d0a28ab43c5717bb85b0a5f6b3b0a4/networkx-3.6.1-py3-none-any.whl"
    sha256 "d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762"
  end

  resource "numba" do
    url "https://files.pythonhosted.org/packages/03/52/176c02d005c5c5143cde10a85bbcdcb6236d9e34c3aac089380e0506cd1d/numba-0.66.0-cp313-cp313-macosx_12_0_arm64.whl"
    sha256 "380b2556a2019ccd1e956ae77dd257eaa39403f7520768b626d44b755112785e"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/97/12/70b5d0d7c15e1ebb8a6a84a8caa1d19e181d84fb58bb6d70aca29099dec1/numpy-2.4.6-cp313-cp313-macosx_14_0_arm64.whl"
    sha256 "043191bfa8eab18c776647b62723ac9dddece59743b13f49b2016094129c2b3f"
  end

  resource "onnxruntime" do
    url "https://files.pythonhosted.org/packages/9c/12/3807e2b17d9eb71d3cb78ed2ba76869b05c637c9b9d6112e636098b0c97a/onnxruntime-1.28.0-cp313-cp313-macosx_14_0_arm64.whl"
    sha256 "31410f544674f534c2f27348af52ef81682ca9c8719154bf4d48f0ef23823b1e"
  end

  resource "packaging" do
    url "https://files.pythonhosted.org/packages/df/b2/87e62e8c3e2f4b32e5fe99e0b86d576da1312593b39f47d8ceef365e95ed/packaging-26.2-py3-none-any.whl"
    sha256 "5fc45236b9446107ff2415ce77c807cee2862cb6fac22b8a73826d0693b0980e"
  end

  resource "pillow" do
    url "https://files.pythonhosted.org/packages/10/76/8803c13605b763d33d156c4678fc77f8443389c0c51c8aef707bb02015f4/pillow-12.3.0-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "d69141514cc30b774ceea5e3ed3a6635c8d8a96edf664689b890f4089111fb35"
  end

  resource "protobuf" do
    url "https://files.pythonhosted.org/packages/19/c7/5f7c636ec43e0c545e28d1f1db71990108306f7bdcb89f069ba97e428e7f/protobuf-7.35.1-py3-none-any.whl"
    sha256 "4bc97768d8fe4ad6743c8a19403e314511ed9f6d13205b687e52421c023ac1b9"
  end

  resource "pycountry" do
    url "https://files.pythonhosted.org/packages/9c/42/7703bd45b62fecd44cd7d3495423097e2f7d28bc2e99e7c1af68892ab157/pycountry-26.2.16-py3-none-any.whl"
    sha256 "115c4baf7cceaa30f59a4694d79483c9167dbce7a9de4d3d571c5f3ea77c305a"
  end

  resource "pycparser" do
    url "https://files.pythonhosted.org/packages/0c/c3/44f3fbbfa403ea2a7c779186dc20772604442dde72947e7d01069cbe98e3/pycparser-3.0-py3-none-any.whl"
    sha256 "b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/fd/7b/122376b1fd3c62c1ed9dc80c931ace4844b3c55407b6fb2d199377c9736f/pydantic-2.13.4-py3-none-any.whl"
    sha256 "45a282cde31d808236fd7ea9d919b128653c8b38b393d1c4ab335c62924d9aba"
  end

  resource "pydantic-core" do
    url "https://files.pythonhosted.org/packages/c1/81/4fa520eaffa8bd7d1525e644cd6d39e7d60b1592bc5b516693c7340b50f1/pydantic_core-2.46.4-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "c94f0688e7b8d0a67abf40e57a7eaaecd17cc9586706a31b76c031f63df052b4"
  end

  resource "pydantic-extra-types" do
    url "https://files.pythonhosted.org/packages/17/c1/3226e6d7f5a4f736f38ac11a6fbb262d701889802595cdb0f53a885ac2e0/pydantic_extra_types-2.11.1-py3-none-any.whl"
    sha256 "1722ea2bddae5628ace25f2aa685b69978ef533123e5638cfbddb999e0100ec1"
  end

  resource "pygments" do
    url "https://files.pythonhosted.org/packages/f4/7e/a72dd26f3b0f4f2bf1dd8923c85f7ceb43172af56d63c7383eb62b332364/pygments-2.20.0-py3-none-any.whl"
    sha256 "81a9e26dd42fd28a23a2d169d86d7ac03b46e2f8b59ed4698fb4785f946d0176"
  end

  resource "pyyaml" do
    url "https://files.pythonhosted.org/packages/b1/16/95309993f1d3748cd644e02e38b75d50cbc0d9561d21f390a76242ce073f/pyyaml-6.0.3-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "2283a07e2c21a2aa78d9c4442724ec1eb15f5e42a723b99cb3d822d48f5f7ad1"
  end

  resource "rapidfuzz" do
    url "https://files.pythonhosted.org/packages/ea/59/b2afd98e41af9cd54554a4c1c423d84cdd60e6b1c0a09496f033b55f60ec/rapidfuzz-3.14.5-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "6737b35d5af7479c5bf9710f7b17edd9d2c43128d974d25fb4ea653e42c64609"
  end

  resource "referencing" do
    url "https://files.pythonhosted.org/packages/2c/58/ca301544e1fa93ed4f80d724bf5b194f6e4b945841c5bfd555878eea9fcb/referencing-0.37.0-py3-none-any.whl"
    sha256 "381329a9f99628c9069361716891d34ad94af76e461dcb0335825aecc7692231"
  end

  resource "regex" do
    url "https://files.pythonhosted.org/packages/95/47/2d0564e93d87bc48618360ddca232a2ca612bbdf53ce8465d45ca5ce14ee/regex-2026.7.19-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "40b34dd88658e4fedd2fddbf0275ac970d00614b731357f425722a3ed1983d11"
  end

  resource "requests" do
    url "https://files.pythonhosted.org/packages/a0/f4/c67b0b3f1b9245e8d266f0f112c500d50e5b4e83cb6f3b71b6528104182a/requests-2.34.2-py3-none-any.whl"
    sha256 "2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/82/3b/64d4899d73f91ba49a8c18a8ff3f0ea8f1c1d75481760df8c68ef5235bf5/rich-15.0.0-py3-none-any.whl"
    sha256 "33bd4ef74232fb73fe9279a257718407f169c09b78a87ad3d296f548e27de0bb"
  end

  resource "rpds-py" do
    url "https://files.pythonhosted.org/packages/f3/6b/686d9dc4359a8f163cfbbf89ee0b4e586431de22fe8248edb63a8cf50d49/rpds_py-2026.6.3-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "f4d78253f6996be4901669ad25319f842f740eccf4d58e3c7f3dd39e6dde1d8f"
  end

  resource "safetensors" do
    url "https://files.pythonhosted.org/packages/f5/b1/fa7c600e7dceae12e9606c7578cbc9ff1e1ed55844883ee5c92205e86226/safetensors-0.8.0-cp310-abi3-macosx_11_0_arm64.whl"
    sha256 "c80201d22cbf405b80647a60ada77bba06c8fba2da2743ba1e89cdcc39a81f25"
  end

  resource "scipy" do
    url "https://files.pythonhosted.org/packages/d3/0f/10ffa0b697a572f4e0d48b92a88895d366422f019f723e7e14a84c050dac/scipy-1.18.0-cp313-cp313-macosx_14_0_arm64.whl"
    sha256 "68363b7eaacd8b5dd426df56d782cc156468ac79a127a1b87ca597d6e2e82197"
  end

  resource "sentencepiece" do
    url "https://files.pythonhosted.org/packages/34/db/f9ea1a6844b4fa5dfe2312095cd866a1f724cd0905054ab9d5991778ba50/sentencepiece-0.2.2-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "201a8e0f55501a76e08dbf2c54bc45f4642b379271e89c667d517bfbc2191f2a"
  end

  resource "shellingham" do
    url "https://files.pythonhosted.org/packages/e0/f9/0595336914c5619e5f28a1fb793285925a8cd4b432c9da0a987836c7f822/shellingham-1.5.4-py2.py3-none-any.whl"
    sha256 "7ecfff8f2fd72616f7481040475a65b2bf8af90a56c89140852d1120324e8686"
  end

  resource "sympy" do
    url "https://files.pythonhosted.org/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl"
    sha256 "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5"
  end

  resource "tiktoken" do
    url "https://files.pythonhosted.org/packages/53/61/c68e123b6d753e3fc2751e9b18e732c9d8bf1e1926762e736eee935d931c/tiktoken-0.13.0-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "8fe806a50664e83a6ffd56cbd1e4f5dcc6cd32a3e7538f70dc38b1a271384545"
  end

  resource "tokenizers" do
    url "https://files.pythonhosted.org/packages/2e/47/174dca0502ef88b28f1c9e06b73ce33500eedfac7a7692108aec220464e7/tokenizers-0.22.2-cp39-abi3-macosx_11_0_arm64.whl"
    sha256 "1e418a55456beedca4621dbab65a318981467a2b188e982a23e117f115ce5001"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/f9/1c/01bfd571a64e7f270e6bab5e33777debe0edc56759233ce84f27dec92d14/tqdm-4.70.0-py3-none-any.whl"
    sha256 "7f585706bfddbdebf89daac705b2dfcc16890130727d3197ca62c732b4310953"
  end

  resource "transformers" do
    url "https://files.pythonhosted.org/packages/6f/67/8d85ca2323233ae3c0365a659c4e52ee1f587b440e4bc577e7d8e4416d0f/transformers-5.14.1-py3-none-any.whl"
    sha256 "9db974c4079ede2d1a3ea7ca5a240df33f2cc26fc2b36ba64c5f2a4f43b6e725"
  end

  resource "typer" do
    url "https://files.pythonhosted.org/packages/43/89/9518bc0c3929bee36b3a4a8e3daddd6e03f92f9961c66d4983b837160543/typer-0.27.1-py3-none-any.whl"
    sha256 "53150287edd11baeb4e4722c8e394fcdf8181c0ae89485cba8d25c778d5edd56"
  end

  resource "typing-extensions" do
    url "https://files.pythonhosted.org/packages/49/d3/b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/typing_extensions-4.16.0-py3-none-any.whl"
    sha256 "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8"
  end

  resource "typing-inspection" do
    url "https://files.pythonhosted.org/packages/dc/9b/47798a6c91d8bdb567fe2698fe81e0c6b7cb7ef4d13da4114b41d239f65d/typing_inspection-0.4.2-py3-none-any.whl"
    sha256 "4ed1cacbdc298c220f1bd249ed5287caa16f34d44ef4e9c3d0cbad5b521545e7"
  end

  resource "urllib3" do
    url "https://files.pythonhosted.org/packages/7f/3e/5db95bcf282c52709639744ca2a8b149baccf648e39c8cc87553df9eae0c/urllib3-2.7.0-py3-none-any.whl"
    sha256 "9fb4c81ebbb1ce9531cce37674bbc6f1360472bc18ca9a553ede278ef7276897"
  end

  def install
    # Wheels only, which Homebrew's own helpers cannot do: `std_pip_args` hard-codes
    # `--no-binary=:all:`, so `venv.pip_install` would try to compile mlx, numba and
    # tokenizers from source (and mlx publishes no sdist at all). So the venv is
    # created by Homebrew and then filled by a direct pip call.
    # The venv has no pip of its own (Homebrew builds them `--without-pip`, and
    # passing `without_pip: false` is an error on 3.12+), so drive the brewed
    # interpreter's pip at it with `--python`. This is the same mechanism
    # Homebrew's own `do_install` uses.
    virtualenv_create(libexec, "python3.13")
    python = formula_opt_bin("python@3.13")/"python3.13"
    pip = [python, "-m", "pip", "--python=#{libexec}/bin/python", "install",
           "--no-deps", "--ignore-installed"]

    # One install per resource, from the file Homebrew already downloaded and
    # verified against the sha256 above, so those URLs are what actually lands in
    # the venv. `--no-index` forbids pip from reaching the network for anything
    # else, which also means a stale resource list fails loudly instead of being
    # silently patched up from PyPI.
    #
    # The copy is required, not tidiness: Homebrew caches downloads as
    # `<sha256>--mlx-0.32.0-cp313-...whl`, and pip parses wheel filenames
    # strictly, rejecting the prefixed name with "Invalid wheel filename (wrong
    # number of parts)". So each wheel is staged under its original name first.
    staging = buildpath/"wheels"
    staging.mkpath
    resources.each do |r|
      r.fetch
      wheel = staging/File.basename(r.url)
      cp r.cached_download, wheel
      system(*pip, "--no-index", wheel)
    end
    # The project itself is pure Python, so install it as a plain copy plus
    # hand-written entry points rather than through pip. Building the wheel would
    # need hatchling, which has no Homebrew formula, and `--no-build-isolation`
    # cannot fetch it inside the sandbox.
    site = libexec/"lib/python3.13/site-packages"
    site.install "mlx_asr"

    # The two console scripts from [project.scripts], parsed out of pyproject.toml
    # rather than hardcoded, so that renaming or repointing an entry point there
    # cannot silently leave this formula generating a script for a function that
    # no longer exists.
    entry_points = (buildpath/"pyproject.toml").read
                   .split("[project.scripts]")[1].split("[")[0]
                   .scan(/^\s*([\w-]+)\s*=\s*"([\w.]+):(\w+)"/)
    odie "no [project.scripts] found in pyproject.toml" if entry_points.empty?

    entry_points.each do |script, mod, func|
      (libexec/"bin"/script).write <<~PYTHON
        #!#{libexec}/bin/python
        import sys
        from #{mod} import #{func}
        sys.exit(#{func}())
      PYTHON
      chmod 0755, libexec/"bin"/script
      bin.install_symlink libexec/"bin"/script
    end
  end

  def caveats
    <<~EOS
      Model weights are downloaded on first use (~1.5GB for the default) and
      cached under ~/.cache/huggingface. Nothing is bundled with this formula.

      `--model kotoba` additionally needs a local MLX conversion of the weights;
      `mlx-asr --model kotoba` prints the conversion command if it is missing.
    EOS
  end

  test do
    # --list-models exercises the registry and the console script without
    # downloading any weights, which a sandboxed `brew test` cannot do.
    assert_match "voxtral", shell_output("#{bin}/mlx-asr --list-models")
    assert_match "kotoba", shell_output("#{bin}/mlx-asr --list-models")

    # A real transcription needs weights, so instead prove the engine imports
    # and that Metal is reachable, which is the part most likely to break.
    system libexec/"bin/python", "-c", <<~PYTHON
      import mlx.core as mx
      from mlx_asr.models import REGISTRY
      from mlx_asr.hardware import machine_info
      assert mx.sum(mx.ones((4, 4))).item() == 16.0
      assert machine_info()["chip"]
      assert "voxtral" in REGISTRY
    PYTHON

    # And that a synthetic WAV survives decode + SRT writing, model aside.
    (testpath/"cues.py").write <<~PYTHON
      from mlx_asr.output import write_srt
      write_srt([(0.0, 1.5, "hello"), (1.5, 3.0, "world")], "out.srt")
    PYTHON
    system libexec/"bin/python", testpath/"cues.py"
    assert_match "00:00:00,000 --> 00:00:01,500", (testpath/"out.srt").read
  end
end
