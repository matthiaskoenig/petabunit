# Release information

## make release
* update release notes in `release-notes` with commit
* make sure all tests run (`tox -p`)
* bump version (`bumpversion [major|minor|patch]`)
* `git push --tags` (triggers release)
* `git push`

* test installation in virtualenv from pypi
```
mkvirtualenv petabunit-test --python=python3.12
(petabunit-test) pip install petabunit
pip list | grep petabunit
deactive petabunit-test
rmvirtualenv petabunit-test
```


