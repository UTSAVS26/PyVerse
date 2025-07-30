"""
TLS certificate parser plugin for PyRecon.
"""

import socket
import ssl
import re
from datetime import datetime
from typing import Dict, Optional, Any, List
from cryptography import x509
from cryptography.hazmat.backends import default_backend


class TLSParser:
    """
    TLS certificate parsing and analysis utility.
    """
    
    def __init__(self, timeout: float = 3.0):
        """
        Initialize the TLS parser.
        
        Args:
            timeout: Connection timeout in seconds
        """
        self.timeout = timeout
        
        # Common certificate authorities
        self.ca_patterns = {
            'Let\'s Encrypt': [r'Let\'s Encrypt', r'LE_'],
            'DigiCert': [r'DigiCert', r'DigiCert Inc'],
            'Comodo': [r'Comodo', r'Sectigo'],
            'GlobalSign': [r'GlobalSign'],
            'GoDaddy': [r'GoDaddy'],
            'Amazon': [r'Amazon'],
            'Google': [r'Google'],
            'Microsoft': [r'Microsoft'],
            'Apple': [r'Apple'],
            'Cloudflare': [r'Cloudflare']
        }
    
    def parse_certificate(self, host: str, port: int = 443) -> Dict[str, Any]:
        """
        Parse TLS certificate from the target.
        
        Args:
            host: Target host
            port: Port to check
            
        Returns:
            Dictionary with certificate information
        """
        result = {
            'subject': {},
            'issuer': {},
            'validity': {},
            'extensions': {},
            'fingerprint': None,
            'serial_number': None,
            'version': None,
            'signature_algorithm': None,
            'public_key_info': {},
            'san': [],
            'ca': None,
            'security_analysis': {}
        }
        
        try:
            # Get certificate
            cert_data = self._get_certificate(host, port)
            if cert_data:
                cert = x509.load_der_x509_certificate(cert_data, default_backend())
                
                # Parse certificate fields
                result['subject'] = self._parse_name(cert.subject)
                result['issuer'] = self._parse_name(cert.issuer)
                result['validity'] = {
                    'not_before': cert.not_valid_before.isoformat(),
                    'not_after': cert.not_valid_after.isoformat()
                }
                result['serial_number'] = str(cert.serial_number)
                result['version'] = cert.version.name
                result['signature_algorithm'] = cert.signature_algorithm_oid._name
                result['fingerprint'] = cert.fingerprint.hex()
                
                # Parse extensions
                result['extensions'] = self._parse_extensions(cert)
                
                # Get SAN
                result['san'] = self._get_san(cert)
                
                # Identify CA
                result['ca'] = self._identify_ca(result['issuer'])
                
                # Security analysis
                result['security_analysis'] = self._analyze_security(cert, result)
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _get_certificate(self, host: str, port: int) -> Optional[bytes]:
        """
        Get certificate in DER format.
        
        Args:
            host: Target host
            port: Port to check
            
        Returns:
            Certificate data or None
        """
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((host, port), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    cert_der = ssock.getpeercert(binary_form=True)
                    return cert_der
                    
        except Exception:
            return None
    
    def _parse_name(self, name) -> Dict[str, str]:
        """
        Parse X.509 name object.
        
        Args:
            name: X.509 name object
            
        Returns:
            Dictionary of name attributes
        """
        result = {}
        
        for attr in name:
            result[attr.oid._name] = attr.value
        
        return result
    
    def _parse_extensions(self, cert) -> Dict[str, Any]:
        """
        Parse certificate extensions.
        
        Args:
            cert: Certificate object
            
        Returns:
            Dictionary of extensions
        """
        extensions = {}
        
        for ext in cert.extensions:
            ext_name = ext.oid._name
            
            if ext_name == 'subjectAltName':
                extensions[ext_name] = self._get_san(cert)
            elif ext_name == 'keyUsage':
                extensions[ext_name] = [usage.name for usage in ext.value]
            elif ext_name == 'extendedKeyUsage':
                extensions[ext_name] = [usage._name for usage in ext.value]
            elif ext_name == 'basicConstraints':
                extensions[ext_name] = {
                    'ca': ext.value.ca,
                    'path_length': ext.value.path_length
                }
            elif ext_name == 'subjectKeyIdentifier':
                extensions[ext_name] = ext.value.hex()
            elif ext_name == 'authorityKeyIdentifier':
                extensions[ext_name] = ext.value.key_identifier.hex()
            else:
                extensions[ext_name] = str(ext.value)
        
        return extensions
    
    def _get_san(self, cert) -> List[str]:
        """
        Get Subject Alternative Names.
        
        Args:
            cert: Certificate object
            
        Returns:
            List of SAN values
        """
        san = []
        
        try:
            san_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            san = [str(name) for name in san_ext.value]
        except Exception:
            pass
        
        return san
    
    def _identify_ca(self, issuer: Dict[str, str]) -> Optional[str]:
        """
        Identify certificate authority.
        
        Args:
            issuer: Issuer information
            
        Returns:
            CA name or None
        """
        issuer_str = ' '.join(issuer.values()).lower()
        
        for ca_name, patterns in self.ca_patterns.items():
            for pattern in patterns:
                if re.search(pattern, issuer_str, re.IGNORECASE):
                    return ca_name
        
        return None
    
    def _analyze_security(self, cert, cert_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze certificate security.
        
        Args:
            cert: Certificate object
            cert_info: Certificate information
            
        Returns:
            Security analysis dictionary
        """
        analysis = {
            'validity_period': 0,
            'key_strength': 0,
            'signature_algorithm': 'unknown',
            'san_present': False,
            'ca_trusted': False,
            'security_score': 0,
            'issues': [],
            'recommendations': []
        }
        
        # Calculate validity period
        not_after = cert.not_valid_after
        not_before = cert.not_valid_before
        validity_days = (not_after - not_before).days
        analysis['validity_period'] = validity_days
        
        # Check key strength
        try:
            key_size = cert.public_key().key_size
            analysis['key_strength'] = key_size
        except Exception:
            pass
        
        # Check signature algorithm
        sig_alg = cert.signature_algorithm_oid._name
        analysis['signature_algorithm'] = sig_alg
        
        # Check SAN
        if cert_info.get('san'):
            analysis['san_present'] = True
        
        # Check CA
        if cert_info.get('ca'):
            analysis['ca_trusted'] = True
        
        # Calculate security score
        score = 0
        
        # Points for key strength
        if analysis['key_strength'] >= 2048:
            score += 30
        elif analysis['key_strength'] >= 1024:
            score += 20
        else:
            score += 10
        
        # Points for signature algorithm
        if 'sha256' in sig_alg.lower():
            score += 25
        elif 'sha1' in sig_alg.lower():
            score += 10
        
        # Points for SAN
        if analysis['san_present']:
            score += 15
        
        # Points for trusted CA
        if analysis['ca_trusted']:
            score += 20
        
        # Points for reasonable validity period
        if 30 <= validity_days <= 365:
            score += 10
        elif validity_days > 365:
            score += 5
        
        analysis['security_score'] = min(score, 100)
        
        # Identify issues
        issues = []
        recommendations = []
        
        if analysis['key_strength'] < 2048:
            issues.append("Weak key size")
            recommendations.append("Use 2048-bit or larger key")
        
        if 'sha1' in sig_alg.lower():
            issues.append("Weak signature algorithm")
            recommendations.append("Use SHA-256 or stronger")
        
        if not analysis['san_present']:
            issues.append("No Subject Alternative Names")
            recommendations.append("Include SAN in certificate")
        
        if not analysis['ca_trusted']:
            issues.append("Untrusted certificate authority")
            recommendations.append("Use certificate from trusted CA")
        
        if validity_days > 365:
            issues.append("Long validity period")
            recommendations.append("Use shorter validity period")
        
        analysis['issues'] = issues
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def get_certificate_chain(self, host: str, port: int = 443) -> List[Dict[str, Any]]:
        """
        Get full certificate chain.
        
        Args:
            host: Target host
            port: Port to check
            
        Returns:
            List of certificates in chain
        """
        chain = []
        
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((host, port), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    # Get certificate chain
                    cert_chain = ssock.getpeercert()
                    
                    if cert_chain:
                        # Parse each certificate in the chain
                        for cert_data in cert_chain:
                            if isinstance(cert_data, tuple):
                                # Handle tuple format
                                cert_info = {
                                    'subject': dict(x[0] for x in cert_data[0]),
                                    'issuer': dict(x[0] for x in cert_data[1]),
                                    'version': cert_data[2],
                                    'serial_number': cert_data[3],
                                    'not_before': cert_data[4],
                                    'not_after': cert_data[5]
                                }
                                chain.append(cert_info)
                    
        except Exception as e:
            chain.append({'error': str(e)})
        
        return chain
    
    def check_certificate_validity(self, host: str, port: int = 443) -> Dict[str, Any]:
        """
        Check certificate validity and expiration.
        
        Args:
            host: Target host
            port: Port to check
            
        Returns:
            Validity check results
        """
        result = {
            'valid': False,
            'expired': False,
            'not_yet_valid': False,
            'days_until_expiry': 0,
            'days_since_issue': 0,
            'expiry_date': None,
            'issue_date': None
        }
        
        try:
            cert_info = self.parse_certificate(host, port)
            
            if 'validity' in cert_info:
                from datetime import timezone
                not_before = datetime.fromisoformat(cert_info['validity']['not_before'])
                not_after = datetime.fromisoformat(cert_info['validity']['not_after'])
                now = datetime.now(timezone.utc)
                
                result['issue_date'] = not_before.isoformat()
                result['expiry_date'] = not_after.isoformat()
                result['days_since_issue'] = (now - not_before).days
                result['days_until_expiry'] = (not_after - now).days
                
                if now < not_before:
                    result['not_yet_valid'] = True
                elif now > not_after:
                    result['expired'] = True
                else:
                    result['valid'] = True
                    
        except Exception as e:
            result['error'] = str(e)
        
        return result 