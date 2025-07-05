apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: mimo-service-account-cluster-role
rules:
- apiGroups: [""]
  #
  # at the HTTP level, the name of the resource for accessing Secret
  # objects is "secrets"
  resources: ["configmaps"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods", "deployments", "jobs"]
  verbs: ["get", "list", "watch", "create", "delete", "patch", "update"]

- apiGroups: ["apiextensions.k8s.io/v1"]
  resources: ["fluidosdeployments"]
  verbs: ["get", "list", "watch", "update", "patch", "delete"]
