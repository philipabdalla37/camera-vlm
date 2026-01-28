- To test that the code works, run it with the environment:
    - myenv\Scripts\activate
    
- Tensorflow has a path that's very long, so during installation, it may most likely crash.
    - Do this on windows:
        - WIN + R -> regedit -> Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem -> Click on "LongPathsEnabled" -> change value to 1
        OR
        - Put project on a smaller path
    - Linux should not have this problem.

- Issue with PyTesseract:
    - If we use PyTesseract, the program needs to be installed first:
        - https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file