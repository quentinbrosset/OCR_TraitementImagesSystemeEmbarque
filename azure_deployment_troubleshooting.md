# Guide de Résolution - Erreur 409 Déploiement Azure

## Problème Identifié
Erreur 409 (Conflict) lors du déploiement avec `azure/webapps-deploy@v3` et OneDeploy.

## Solutions Recommandées (par ordre de priorité)

### 1. **Solution Immédiate - Redémarrer le Service**
```bash
# Via Azure CLI (si installé)
az webapp restart --name SegmentationImages --resource-group [votre-resource-group]

# Ou via le portail Azure :
# Azure Portal > App Services > SegmentationImages > Redémarrer
```

### 2. **Modifier le Workflow GitHub Actions**

#### Option A : Ajouter un délai et une stratégie de retry
```yaml
- name: 'Deploy to Azure Web App'
  id: deploy-to-webapp
  uses: azure/webapps-deploy@v3
  with:
    app-name: ${{ env.AZURE_WEBAPP_NAME }}
    publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
    package: deployment.zip
    clean: true
    restart: true
    timeout: 600000  # 10 minutes
  continue-on-error: true

- name: 'Retry deployment if failed'
  if: steps.deploy-to-webapp.outcome == 'failure'
  uses: azure/webapps-deploy@v3
  with:
    app-name: ${{ env.AZURE_WEBAPP_NAME }}
    publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
    package: deployment.zip
    clean: true
    restart: true
```

#### Option B : Arrêter l'app avant le déploiement
```yaml
- name: 'Stop Azure Web App'
  run: |
    # Cette étape nécessiterait Azure CLI configuré
    # Ou utiliser l'API REST Azure
    echo "Stopping app before deployment"

- name: 'Deploy to Azure Web App'
  id: deploy-to-webapp
  uses: azure/webapps-deploy@v3
  with:
    app-name: ${{ env.AZURE_WEBAPP_NAME }}
    publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
    package: deployment.zip
    clean: true
    restart: true

- name: 'Start Azure Web App'
  run: |
    echo "Starting app after deployment"
```

### 3. **Vérifications et Nettoyage**

#### A. Vérifier l'état du déploiement
- Allez sur le portail Azure
- App Services > SegmentationImages > Deployment Center
- Vérifiez s'il y a un déploiement en cours ou échoué

#### B. Nettoyer les fichiers temporaires
- Dans le portail Azure : App Services > Advanced Tools (Kudu) > Debug console
- Naviguez vers `/home/site/wwwroot` et supprimez les fichiers temporaires

### 4. **Optimisations du Workflow**

#### Modifier la section de création du package :
```yaml
- name: Zip application for deployment
  run: |
    # Nettoyer d'abord
    rm -rf deploy_package deployment.zip
    
    mkdir -p deploy_package
    cp -r application deploy_package/
    cp requirements.txt deploy_package/
    cp startup.sh deploy_package/
    
    # Ajouter un .deployment pour Azure
    cat > deploy_package/.deployment << 'EOF'
    [config]
    command = startup.sh
    EOF
    
    # Créer un web.config optimisé
    cat > deploy_package/web.config << 'EOF'
    <?xml version="1.0" encoding="utf-8"?>
    <configuration>
      <system.webServer>
        <handlers>
          <add name="PythonHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified"/>
        </handlers>
        <httpPlatform processPath="%HOME%\site\wwwroot\startup.sh"
                      arguments=""
                      stdoutLogEnabled="true"
                      stdoutLogFile="%HOME%\LogFiles\stdout.log"
                      startupTimeLimit="120"
                      requestTimeout="00:04:00">
          <environmentVariables>
            <environmentVariable name="PYTHONPATH" value="%HOME%\site\wwwroot" />
          </environmentVariables>
        </httpPlatform>
      </system.webServer>
    </configuration>
    EOF
    
    cd deploy_package
    zip -r ../deployment.zip . -x "*.pyc" "*/__pycache__/*"
    cd ..
```

### 5. **Actions Alternatives**

#### Option 1 : Utiliser Azure CLI Deploy
```yaml
- name: Deploy with Azure CLI
  run: |
    az webapp deployment source config-zip \
      --resource-group [votre-resource-group] \
      --name SegmentationImages \
      --src deployment.zip
```

#### Option 2 : Utiliser FTP Deploy
```yaml
- name: Deploy via FTP
  uses: SamKirkland/FTP-Deploy-Action@v4.3.4
  with:
    server: ${{ secrets.FTP_SERVER }}
    username: ${{ secrets.FTP_USERNAME }}
    password: ${{ secrets.FTP_PASSWORD }}
    local-dir: deploy_package/
```

## Actions Immédiates Recommandées

1. **Redémarrez votre App Service** via le portail Azure
2. **Attendez 5-10 minutes** avant de relancer le déploiement
3. **Vérifiez les logs** dans App Services > Log stream
4. **Essayez le déploiement avec la stratégie de retry**

## Monitoring et Debugging

- **Logs en temps réel** : App Services > Log stream
- **Historique des déploiements** : App Services > Deployment Center
- **Kudu console** : App Services > Advanced Tools > Go

## Configuration Recommandée pour l'App Service

Dans Configuration > General settings :
- **Stack** : Python
- **Major version** : 3.11
- **Startup Command** : `/home/site/wwwroot/startup.sh`
- **Always On** : On (si pas sur un plan gratuit)

La cause la plus probable de votre erreur 409 est un déploiement précédent qui n'est pas complètement terminé ou des fichiers verrouillés. Un simple redémarrage du service devrait résoudre le problème dans la plupart des cas.