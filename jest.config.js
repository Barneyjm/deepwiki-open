// jest.config.js
module.exports = {
  testEnvironment: 'jest-environment-node', // or 'node'
  roots: ['<rootDir>/src'],
  testPathIgnorePatterns: ['/node_modules/', '/.next/'],
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', { tsconfig: 'tsconfig.json' }],
  },
  moduleNameMapper: {
    // Handle CSS imports (if any)
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    // Handle module path aliases
    '^@/components/(.*)$': '<rootDir>/src/components/$1',
    '^@/utils/(.*)$': '<rootDir>/src/utils/$1',
    '^@/(.*)$': '<rootDir>/src/$1', // Generic alias based on tsconfig
    // Add any other aliases from tsconfig.json
  },
  // setupFilesAfterEnv: ['<rootDir>/jest.setup.js'], // Optional: if we need setup files
  // globals: {}, // Or remove globals if no longer needed for other things
};
