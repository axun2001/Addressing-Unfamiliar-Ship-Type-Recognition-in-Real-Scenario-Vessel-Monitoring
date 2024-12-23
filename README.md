# Addressing-Unfamiliar-Ship-Type-Recognition-in-Real-Scenario-Vessel-Monitoring

This repository contains the code accompanying the paper:

**"Addressing Unfamiliar Ship Type Recognition in Real-Scenario Vessel Monitoring: A Multi-Angle Metric Networks Framework"**

## Introduction

This repository provides a solution to the problem of recognizing and detecting unknown types of ships in real-world environments by comparing the similarity of each set of image data using the ternary twin network described in the paper. The method proposed in this paper aims to efficiently detect different types of ship examples and can be applied to the Pytorch framework.

## Repository Structure

- **train_script.py**: The main training portal of the program.
- **Single_prediction_script.py**: Contains the proposed method for detecting similarity between individual samples, which can be adapted to other backbone network models.
- **Batch_prediction_script.py**: Contains the proposed method for detecting similarity between multiple samples, which can be adapted to other backbone network models.

 ## Runtime Environment Configuration

The experiments were performed on a computer including a 64-bit Windows 11 operating system, a 12th generation Intel Core i7-12700 processor, 32 GB of RAM, and an NVIDIA GeForce RTX 3060. The PyTorch deep learning framework with version 11.7 was used, PyCharm as the main software compilation tool and Python 3.9 as the programming language.

## Example Dataset Source

The repository contains the datasets accompanying the paper available at this link: **[ShipMonitoring-LSS Dataset](https://doi.org/10.6084/m9.figshare.27874146)**.
