cpu-only environment containing deepspparks code. Used for the "getting started" demo.

Contains 2 versions:
  - main is used for running mlflow projects
  - dex (data exploration) contains a jupyter environment and can be used for data analysis.

To build the container for mlflow projects: 
```bash
./container -b
```

To build the container for jupyter environments:
```bash
./container -b --dex
```

To run the container, see the `getting started` section of Deepspparks.
