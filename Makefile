.PHONY: box-setup box-check

box-setup:
	bash scripts/setup_aws_box.sh

box-check:
	bash scripts/verify_gpu.sh
