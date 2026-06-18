# Week 8 Warmup

## Cloud Concepts

### Question 1

The core economic model of cloud computing is pay-as-you-go: instead of buying and maintaining your own servers, you rent compute, storage, and networking resources from a cloud provider and pay for what you actually use. Owning your own servers requires paying upfront for hardware, maintenance, upgrades, and extra capacity even when it sits idle.

### Question 2

Vertical scaling means making one machine more powerful, for example moving a model training job to a machine with more RAM and a faster GPU. Horizontal scaling means adding more machines and splitting the work across them, for example adding more web servers when traffic suddenly grows.

1. A web app that suddenly needs to handle 100,000 users after a viral launch needs horizontal scaling because the extra traffic can be spread across more server instances.

2. A data scientist who needs a faster GPU and more RAM for one model training job needs vertical scaling because they are making one machine more powerful.

3. A data pipeline that now needs to process 10,000 files needs horizontal scaling because the files can be split across multiple machines and processed in parallel.

### Question 3

- Gmail: SaaS. I just use the finished email application in a browser, and Google manages the servers, storage, updates, and application code.

- Azure Virtual Machines: IaaS. Azure provides the virtual machine infrastructure, but I am responsible for choosing the OS, installing software, configuring the environment, and managing updates.

- Azure App Service: PaaS. I deploy my application code, while Azure manages the underlying servers, scaling, and platform infrastructure.

- AWS S3: IaaS. It provides cloud storage infrastructure where I manage buckets, files, permissions, and usage, but I do not manage the physical storage hardware.

- GitHub Codespaces: PaaS. It gives me a ready-to-use development environment in the cloud, so I mainly manage my code and project configuration rather than the underlying machine.

- Snowflake: SaaS. It is a managed data platform where I use the warehouse/database features, while Snowflake manages much of the infrastructure, scaling, and maintenance.

IaaS, or Infrastructure as a Service, means renting basic cloud infrastructure such as virtual machines, storage, and networking. An example is Azure Virtual Machines. As the developer, I manage the operating system, installed software, security updates, and how my application runs.

PaaS, or Platform as a Service, means the provider manages the infrastructure and runtime platform, while I focus on deploying and configuring my code. An example is Azure App Service. As the developer, I manage my application code, dependencies, and configuration, but not the underlying servers.

SaaS, or Software as a Service, means using a complete application that someone else builds and operates. An example is Gmail. As the user or developer, I manage very little beyond my account, settings, and data inside the application.

### Question 4

A managed data platform like Databricks or Snowflake is a higher-level platform built for data and analytics work, often running on top of a cloud provider like Azure, AWS, or GCP. Instead of assembling compute, storage, networking, and data tools yourself, the platform gives you a more ready-made environment for querying, processing, and analyzing data.

Compared with using Azure directly, you gain speed and convenience because many pieces are already connected and managed for you. You give up some flexibility and control over the exact infrastructure, and the platform may cost more than building the same stack yourself directly on the cloud provider.

### Question 5

The cloud may not be the right choice when the dataset fits comfortably on one local machine and the compute needs are small, because local processing can be faster and cheaper. It may also be the wrong choice for an early prototype if the cloud setup adds more complexity than value.

Another practical concern is cost: if resources are left running or storage is not cleaned up, cloud bills can grow quickly.

## Azure Basics

### Question 1

An Azure subscription is the billing account that owns the cloud resources. In this course, the subscription is shared by CTD.

A resource group is a container for related Azure resources inside a subscription. My personal resource group is my own sandbox for the course, while CTD owns and manages the shared subscription.

### Question 2

Azure Cloud Shell is ephemeral by default, which means files and folders created during a shell session may be deleted when the session ends or restarts. For this course, Cloud Shell is connected to a persistent Azure file share, so files in the home directory, SSH keys, scripts, and configuration can survive between sessions.

### Question 3

The SSH private key is the secret key that stays with me and should never be shared. The SSH public key is the key that can be uploaded to remote systems I want to connect to.

Uploading the public key is safe because it cannot be used to recreate the private key. When I connect, SSH checks that my private key matches the public key on the remote system, so I can prove my identity without sending a password or sharing the private key.

### Question 4

```json
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#uevgenya18@outlook.com",
    "type": "user"
  }
}
```

Adding `--output table` shows the same account information in a more readable table format instead of raw JSON. The table output is easier to scan quickly, while JSON is more complete and better for scripts.