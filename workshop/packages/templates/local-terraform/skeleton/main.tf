provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "rg" {
  name     = "${projectName}-rg"
  location = "${region}"
}
