# Week 8 Project

## Video Link

[Video link here.](https://drive.google.com/file/d/1Wl7M2WYQI3TRkNoH3aaO1NPoSvZR2eoo/view?usp=sharing)

## Cost Analysis Summary

### Scenario A: Lightweight Compute

Scenario A uses a Standard_B1s Linux virtual machine in East US for 160 hours per month. The Azure Pricing Calculator estimated this at about $1.66 per month. This was much cheaper than I expected, but it makes sense because the VM is very small and only runs part time.

### Scenario B: Heavy Analytics Workload

Scenario B includes a Standard_NC6s_v3 GPU virtual machine running all month, an Azure SQL Database in the General Purpose tier with 4 vCores, and a Blob Storage account with about 1 TB of data. My estimate was:

- GPU VM: $2,233.80 per month
- Azure SQL Database: $449.26 per month
- Blob Storage: $21.84 per month

The total for Scenario B was about $2,704.90 per month. The GPU VM was the biggest cost by far, which was the most interesting part of the estimate to me.

## Pricing Calculator Notes

The biggest surprise was how different the two compute scenarios were. The small B1s VM was only a few dollars per month, while the GPU VM was over two thousand dollars per month when running 24/7. This made it clear why it is important to shut down expensive cloud resources when they are not needed.

## Cloud Shell Script Output

When I ran `project_08.py`, it printed:

```text
=== Monthly Cost Estimates ===
Scenario A (lightweight):       $1.60
Scenario B (GPU VM only):       $2233.80
Scenario B VM costs 1396.1x more than Scenario A
```
The Scenario B GPU VM cost matched the Pricing Calculator. Scenario A was slightly lower in the script because the calculator displayed the B1s hourly rate rounded to `$0.010/hour`, while the calculator's monthly total appears to use a more precise rate internally.
