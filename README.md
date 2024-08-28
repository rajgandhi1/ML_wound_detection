# ML Wound Detection

This repository hosts a machine learning project aimed at detecting wounds, ulcers, and calluses from images using a pre-trained GroundingDINO model. The project is built with Azure Functions to facilitate deployment and scaling in cloud environments. The following README provides a comprehensive guide to setting up, using, and deploying the project.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contribution](#contribution)
- [License](#license)

## Project Overview
This project leverages the GroundingDINO model to identify wounds, ulcers, and calluses in images. The model is integrated into an Azure Function to enable easy and scalable deployment. The function accepts an image URL as input and returns the processed image with detected regions highlighted.

## Getting Started

### Prerequisites
- **Python 3.8+**
- **Azure Functions Core Tools**
- **Visual Studio Code** with the Azure Functions extension for development and deployment.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rajgandhi1/ML_wound_detection.git
   cd ML_wound_detection
2. Install the required Python packages:
   ```bash
   python -m pip install -r requirements.txt
3. Set up the Azure Function project:
   - Follow the [Azure Functions Quickstart](https://aka.ms/azure-functions/python/quickstart) guide to set up the function environment.
   - The project is configured to work with Python using Visual Studio Code.

## Directory Structure

The repository is organized as follows:

```plaintext
ML_wound_detection/
│
├── .vscode/
│   ├── extensions.json       # Recommended VS Code extensions
│   ├── launch.json           # Debugging configurations
│   ├── settings.json         # VS Code settings for Azure Functions
│   └── tasks.json            # Automation tasks for Azure Functions
│
├── lucent/
│   ├── __init__.py           # Main Python file for the Azure Function
│   ├── function.json         # Binding configurations for the function
│   └── host.json             # Hosting configuration for the function
│
├── getting_started.md        # Instructions for getting started with Azure Functions
├── host.json                 # Global configuration options for Azure Functions
├── requirements.txt          # List of Python dependencies
└── .gitmodules               # Information about submodules used (GroundingDINO)
```

### Key Files and Directories

- **`lucent/__init__.py`**: Contains the core logic for the Azure Function, including loading the model, processing the image, and returning the annotated result.
- **`requirements.txt`**: Lists all the necessary Python libraries for the project.
- **`.gitmodules`**: Specifies the GroundingDINO submodule, which is essential for the model's operation.

## Dependencies

The project relies on several Python libraries, including:
- `azure-functions`
- `requests`
- `numpy`
- `torch`
- `opencv-python`
- `Pillow`
- `GroundingDINO` (included as a submodule)

Ensure all dependencies are installed using the `requirements.txt` file.

## Usage

### Running the Function Locally

1. Start the Azure Function locally:
   ```bash
   func start
2. Test the function by sending a request with an image URL:
   ```bash
   curl -X GET "http://localhost:7071/api/HttpTrigger?imageUrl=YOUR_IMAGE_URL"

## Testing

The project is set up to support unit testing. Place your test cases inside the `tests/` directory and use your preferred testing framework.

## Deployment

### Deploying to Azure

1. Ensure you have the Azure CLI and Azure Functions Core Tools installed.

2. Log in to your Azure account:

   ```bash
   az login
3. Deploy the function:

   ```bash
   func azure functionapp publish <YOUR_FUNCTION_APP_NAME>
Refer to the Azure Functions Developer Guide for more detailed instructions.

## Contribution

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code follows the project's coding standards and includes tests where appropriate.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
