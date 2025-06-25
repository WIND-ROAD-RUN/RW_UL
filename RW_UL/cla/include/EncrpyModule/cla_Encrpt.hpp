#pragma once

#include <string>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace rw
{
	namespace cla
	{
		class SymmetricEncryptor
		{
		public:
			// Converts binary data to a hexadecimal string
			static std::string toHex(const std::vector<unsigned char>& data)
			{
				std::ostringstream oss;
				for (unsigned char byte : data)
				{
					oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
				}
				return oss.str();
			}

			// Encrypts the input string using the provided key
			static std::string encrypt(const std::string& plaintext, const std::string& key)
			{
				// Initialization vector (IV) for AES
				unsigned char iv[EVP_MAX_IV_LENGTH];
				if (!RAND_bytes(iv, sizeof(iv)))
				{
					throw std::runtime_error("Failed to generate IV");
				}

				// Prepare encryption context
				EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
				if (!ctx)
				{
					throw std::runtime_error("Failed to create encryption context");
				}

				std::vector<unsigned char> ciphertext(plaintext.size() + EVP_MAX_BLOCK_LENGTH);
				int len = 0, ciphertext_len = 0;

				try
				{
					// Initialize encryption operation
					if (EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, reinterpret_cast<const unsigned char*>(key.data()), iv) != 1)
					{
						throw std::runtime_error("Failed to initialize encryption");
					}

					// Encrypt the plaintext
					if (EVP_EncryptUpdate(ctx, ciphertext.data(), &len, reinterpret_cast<const unsigned char*>(plaintext.data()), plaintext.size()) != 1)
					{
						throw std::runtime_error("Failed to encrypt data");
					}
					ciphertext_len = len;

					// Finalize encryption
					if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + len, &len) != 1)
					{
						throw std::runtime_error("Failed to finalize encryption");
					}
					ciphertext_len += len;

					// Append IV to the ciphertext
					ciphertext.insert(ciphertext.end(), iv, iv + sizeof(iv));
				}
				catch (...)
				{
					EVP_CIPHER_CTX_free(ctx);
					throw;
				}

				EVP_CIPHER_CTX_free(ctx);

				// Convert ciphertext to hexadecimal
				return toHex(ciphertext);
			}

			// Decrypts the input string using the provided key
			static std::string decrypt(const std::string& hexCiphertext, const std::string& key)
			{
				// Convert hexadecimal string back to binary
				std::vector<unsigned char> ciphertext;
				for (size_t i = 0; i < hexCiphertext.size(); i += 2)
				{
					std::string byteString = hexCiphertext.substr(i, 2);
					unsigned char byte = static_cast<unsigned char>(std::stoi(byteString, nullptr, 16));
					ciphertext.push_back(byte);
				}

				// Extract IV from the end of the ciphertext
				if (ciphertext.size() < EVP_MAX_IV_LENGTH)
				{
					throw std::runtime_error("Ciphertext too short");
				}
				unsigned char iv[EVP_MAX_IV_LENGTH];
				std::copy(ciphertext.end() - EVP_MAX_IV_LENGTH, ciphertext.end(), iv);

				// Prepare decryption context
				EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
				if (!ctx)
				{
					throw std::runtime_error("Failed to create decryption context");
				}

				std::vector<unsigned char> plaintext(ciphertext.size());
				int len = 0, plaintext_len = 0;

				try
				{
					// Initialize decryption operation
					if (EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, reinterpret_cast<const unsigned char*>(key.data()), iv) != 1)
					{
						throw std::runtime_error("Failed to initialize decryption");
					}

					// Decrypt the ciphertext
					if (EVP_DecryptUpdate(ctx, plaintext.data(), &len, ciphertext.data(), ciphertext.size() - EVP_MAX_IV_LENGTH) != 1)
					{
						throw std::runtime_error("Failed to decrypt data");
					}
					plaintext_len = len;

					// Finalize decryption
					if (EVP_DecryptFinal_ex(ctx, plaintext.data() + len, &len) != 1)
					{
						throw std::runtime_error("Failed to finalize decryption");
					}
					plaintext_len += len;
				}
				catch (...)
				{
					EVP_CIPHER_CTX_free(ctx);
					throw;
				}

				EVP_CIPHER_CTX_free(ctx);
				return std::string(plaintext.begin(), plaintext.begin() + plaintext_len);
			}
		};
	}
}