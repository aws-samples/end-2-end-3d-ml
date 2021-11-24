# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

#!/bin/bash
aws cloudformation deploy \
    --template cloudformation.yaml \
    --stack-name threedee \
    --capabilities CAPABILITY_NAMED_IAM \
#     --parameter-overrides ShouldCreateBucketInputParameter=False